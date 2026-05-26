#!/bin/sh
cd "$(dirname $(readlink -f "$0"))/.."

# stashName=`git stash create`;
git archive --format=zip -o "pystachio_src.zip" HEAD
