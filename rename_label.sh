#! /usr/bin/bash
for i in `ls`; do mv $i `echo $i | sed 's/\-/\>/g'`; done
