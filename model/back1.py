import sys
import os


a,ran = map(int, sys.stdin.readline().split())

lit = [i+1 for i in range(a)]
for i in range(0,ran):
    x,y = map(int, sys.stdin.readline().split())
    lit[x-1],lit[y-1] = lit[y-1],lit[x-1]

print(*lit)# 리스트 값만 출력