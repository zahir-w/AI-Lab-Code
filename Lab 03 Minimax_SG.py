import minimax
import pickle as pkl

sg = [0 for i in range(201)]
son = [[]]
for i in range(1, 201):
    if i < 4:
        son.append([i - 1])
    elif 4 <= i < 7:
        son.append([i - 1, i - 4])
    else:
        son.append([i - 1, i - 4, i - 7])

minimax.SG(sg, son)
f = open('Variables_Q4.pkl', 'wb')
if sg[100] == 1:
    print("A wins!")
    pkl.dump({'Q4_i': 'A', 'Q4_ii': sg}, f)
else:
    print("B wins!")
    pkl.dump({'Q4_i': 'B', 'Q4_ii': sg}, f)
f.close()
print(sg)
