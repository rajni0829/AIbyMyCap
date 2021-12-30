# Input: list1 = [12, -7, 5, 64, -14] Output: 12, 5, 64 Input: list2 = [12, 14, -95, 3] Output: [12, 14, 3]

list1 = [12, -7, 5, 64, -14]
list2 = [12, 14, -95, 3]

for num in range(len(list1)-1):
    if list1[num] < 0:
        list1.pop(num)

for num in range(len(list2)-1):
    if list2[num] < 0:
        list2.pop(num)

print(list1)
print(list2)
