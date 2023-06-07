import pickle
import numpy

with open('successful_faces.pkl', 'rb') as file:
    arr = numpy.array(pickle.load(file))

print(90 - numpy.max(arr)) #19.95 70
print(90 - numpy.min(arr)) #14.50 76