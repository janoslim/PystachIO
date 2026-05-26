from enum import Enum
import torch
import torch.nn as nn
import datetime
from helpers import util

# Filter operation for pruning and filtering

class FilterOp(Enum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    II = "ii"
    NI = "ni"

class FilterOpCol(Enum):
    EQC = "eqc"
    NEC = "nec"
    LTC = "ltc"
    LEC = "lec"
    GTC = "gtc"
    GEC = "gec"

class FilterCondition:
    def __init__(self, op: FilterOp, value):
        self.op = op
        self.value = value
        self.op_value = value
        if self.op == FilterOp.II or self.op == FilterOp.NI:
            self.value = torch.min(value)       # IsIn's min 
            self.extra_value = torch.max(value) #  IsIn's max
    
    def __repr__(self):
        return f"FilterCondition({self.op.value} {self.value} with op_value={self.op_value})"
    
    def to_tensor_lambda(self):
        # Parquet adapts date to int32
        if isinstance(self.value, datetime.date):
            self.op_value = util.date_to_epoch_days(self.value.year, self.value.month, self.value.day)
        if self.op == FilterOp.EQ:
            return lambda tensor: torch.eq(tensor, self.op_value)
        elif self.op == FilterOp.NE:
            return lambda tensor: torch.ne(tensor, self.op_value)
        elif self.op == FilterOp.LT:
            return lambda tensor: torch.lt(tensor, self.op_value)
        elif self.op == FilterOp.LE:
            return lambda tensor: torch.le(tensor, self.op_value)
        elif self.op == FilterOp.GT:
            return lambda tensor: torch.gt(tensor, self.op_value)
        elif self.op == FilterOp.GE:
            return lambda tensor: torch.ge(tensor, self.op_value)
        elif self.op == FilterOp.II:
            return lambda tensor: torch.isin(tensor, self.op_value)
        elif self.op == FilterOp.NI:
            return lambda tensor: torch.logical_not(torch.isin(tensor, self.op_value))
        elif self.op == FilterOpCol.EQC:
            return lambda tensora,tensorb: torch.eq(tensora, tensorb)
        elif self.op == FilterOpCol.NEC:
            return lambda tensora,tensorb: torch.ne(tensora, tensorb)
        elif self.op == FilterOpCol.LTC:
            return lambda tensora,tensorb: torch.lt(tensora, tensorb)
        elif self.op == FilterOpCol.LEC:
            return lambda tensora,tensorb: torch.le(tensora, tensorb)
        elif self.op == FilterOpCol.GTC:
            return lambda tensora,tensorb: torch.gt(tensora, tensorb)
        elif self.op == FilterOpCol.GEC:
            return lambda tensora,tensorb: torch.ge(tensora, tensorb)
        else:
            raise ValueError(f"Unsupported filter operation: {self.op}")
    
    def evaluate_on_range(self, min_val, max_val):
        """
        For row group **pruning**, check if a range [min_val, max_val] could potentially satisfy this filter.
        Returns:
            bool: True if the range **might** contain values that satisfy the filter
        """
        try:
            if self.op == FilterOp.EQ:
                return min_val <= self.value <= max_val
            elif self.op == FilterOp.NE:
                return min_val != self.value or max_val != self.value
            elif self.op == FilterOp.LT:
                return min_val < self.value
            elif self.op == FilterOp.LE:
                return min_val <= self.value
            elif self.op == FilterOp.GT:
                return max_val > self.value
            elif self.op == FilterOp.GE:
                return max_val >= self.value
            elif self.op == FilterOp.II:
                # [min_val,   ..., self.value<->self.extra_value , ...  , max_val] => IsIn
                return min_val <= self.value and self.extra_value <= max_val
            elif self.op == FilterOp.NI:
                return True
        except Exception:
            raise ValueError("Cannot evaluate range with given filter value type")
    
    @staticmethod
    def combine_condition_to_lambda(condition_list, column_idx):
        """Combine multiple filter condition_list for the same column into a single lambda using AND logic"""
        # assert len(condition_list) != 0
        if len(condition_list) == 1:
            filter_lambda = condition_list[0].to_tensor_lambda()
            return lambda tensor: filter_lambda(tensor)
        
        def combined_lambda(tensor):
            result = torch.full_like(tensor, True, dtype=torch.bool)
            for condition in condition_list:
                condition_lambda = condition.to_tensor_lambda()
                result &= condition_lambda(tensor)
            return result
        return combined_lambda

class TableFilter(nn.Module):
    def __init__(self, column_conditions):
        super().__init__()
        self.column_conditions = column_conditions # Multi-map: column_idx -> list of FilterCondition
        self.column_lambdas = {}
        self.row_lambdas = [] # (cola_idx,colb_idx,lambda taking both tensors)
        for col_idx, conditions in column_conditions.items():
            immediate_conds = list(filter(lambda cond: isinstance(cond.op, FilterOp), conditions)) # all immediate conditions -> these will be combined
            multi_conds = list(filter(lambda cond: isinstance(cond.op, FilterOpCol), conditions)) # conditions like col1 < col2

            self.column_lambdas[col_idx] = FilterCondition.combine_condition_to_lambda(immediate_conds, col_idx)
            self.row_lambdas += [(col_idx, cond.value, cond.to_tensor_lambda()) for cond in multi_conds]
            column_conditions[col_idx] = immediate_conds # overwrite, keeping only the immediate conds (used for pruning)

    def forward(self, child):
        data = child()
        if not self.column_lambdas:
            return data
        mask = torch.full((data[0].shape[0],), True, dtype=torch.bool, device=data[0].device)
        for col_idx, filter_lambda in self.column_lambdas.items():
            if filter_lambda is not None:
                column_mask = filter_lambda(data)
                mask &= column_mask
        return [torch.masked_select(col, mask) for col in data]

if __name__ == '__main__':
    data = [
        torch.arange(10),
        torch.arange(100, 110),
        torch.tensor([9204, 9205, 9206, 9207, 9208, 9209, 9210, 9211, 9212, 9213])
    ]
    print(f"Original data:")
    print(f"  Column 0: {data[0]}")
    print(f"  Column 1: {data[1]}")
    print(f"  Column 2 (dates): {data[2]}")
    
    # Single filter: values >= 5 in column 0
    filter1 = FilterCondition(FilterOp.GE, 5)
    mask1 = filter1.to_tensor_lambda()(data[0])
    result1 = [torch.masked_select(col, mask1) for col in data]
    print(f"\nFilter: column 0 >= 5")
    print(f"  Filtered column 0: {result1[0]}")
    print(f"  Filtered column 1: {result1[1]}")
    print(f"  Filtered column 2: {result1[2]}")
    
    # Multiple filters: 3 <= values < 7 in column 0
    filters = [FilterCondition(FilterOp.GE, 3), FilterCondition(FilterOp.LT, 7)]
    combined = FilterCondition.combine_condition_to_lambda(filters, 0)
    mask2 = combined(data[0])
    result2 = [torch.masked_select(col, mask2) for col in data]
    print(f"\nFilter: 3 <= column 0 < 7")
    print(f"  Filtered column 0: {result2[0]}")
    print(f"  Filtered column 1: {result2[1]}")
    print(f"  Filtered column 2: {result2[2]}")
    
    # Date filter with mask select
    date = datetime.date(1995, 3, 17)
    date_filter = FilterCondition(FilterOp.GT, date)
    date_mask = date_filter.to_tensor_lambda()(data[2])
    date_result = [torch.masked_select(col, date_mask) for col in data]
    print(f"\nDate filter: > {date} (epoch day {util.date_to_epoch_days(1995, 3, 17)})")
    print(f"  Date mask: {date_mask}")
    print(f"  Filtered column 0: {date_result[0]}")
    print(f"  Filtered column 1: {date_result[1]}")
    print(f"  Filtered dates: {date_result[2]}")
    
    # Range check for pruning
    can_satisfy = date_filter.evaluate_on_range(datetime.date(1995, 3, 10), datetime.date(1995, 3, 20))
    print(f"  Range [1995-03-10, 1995-03-20] can satisfy: {can_satisfy}")
