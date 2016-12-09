list1 = [1, 2, 3]
list2 = [x * 2 for x in list1 if x % 2 != 0]
print list2

d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    print animal
    legs = d[animal]
    print 'A %s has %d legs' % (animal, legs)