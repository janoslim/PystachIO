#!/bin/bash

# IPv4 address for enp12s0np0:  192.168.1.20
# IPv4 address for enp18s0np0:  192.168.1.21
# IPv4 address for enp75s0np0:  192.168.1.22
# IPv4 address for enp84s0np0:  192.168.1.23
# IPv4 address for enp141s0np0: 192.168.1.24
# IPv4 address for enp148s0np0: 192.168.1.25
# IPv4 address for enp186s0np0: 192.168.1.26
# IPv4 address for enp204s0np0: 192.168.1.27

for i in {0..7}; do
    ibdev="mlx5_$i"
    netdev=`ibdev2netdev | grep $ibdev | cut -d' ' -f5`
    ip="192.168.1.2$i"
    mac_address=`ip a show $netdev | grep "link/ether" | cut -d' ' -f 6`
    echo "$ibdev maps to $netdev, mac $mac_address. Should get ip=$ip"

    sudo ip addr flush dev "$netdev"
    sudo ip addr add "$ip/24" dev "$netdev"
    # https://transactional.blog/rdma/soft-roce-requires-ipv6-link-local
    sudo ip l set dev "$netdev" addrgenmode eui64
    sudo ip l set dev "$netdev" mtu 9100
    sudo ip l set dev "$netdev" up
done

# fix the arp table, because our node cant seem to do this itself:
for i in {0..7}; do
    ibdev="mlx5_$i"
    netdev=`ibdev2netdev | grep $ibdev | cut -d' ' -f5`
    ip="192.168.1.2$i"
    mac_address=`ip a show $netdev | grep "link/ether" | cut -d' ' -f 6`
    # same variables as above loop
    for j in {0..7}; do
        if [ "$i" -eq "$j" ]; then
            continue
        fi
        other_netdev=`ibdev2netdev | grep mlx5_$j | cut -d' ' -f5`
        # echo "$other_netdev"
        # echo "ip neighbor replace $ip lladdr $mac_address nud permanent dev $other_netdev"
        sudo ip neighbor replace $ip lladdr $mac_address nud permanent dev $other_netdev
    done
done


