import ctypes
libtest = ctypes.cdll.LoadLibrary('./distance/xh_dis.so')

def edit_distance(arr1, arr2):
	return (libtest.get_distance(
		(ctypes.c_int * len(arr1))(*arr1),
		(ctypes.c_int * len(arr2))(*arr2),
		len(arr1),
		len(arr2)))

def get_ed_btwn_lists(arr1, arr2, arr3, arr4):
	'''
	input:
		arr1: a list
		arr2: a list
		m: int, namely length of arr3[]
		n: length of arr4[]
	output:
		result: a list
	usage:
		The example is in main()
	'''
	m = int(len(arr3))
	n = int(len(arr4))
	buffering = (ctypes.c_int * ( m * n ))()
	libtest.get_distance_between_lists(
			(ctypes.c_int * len(arr1)) (*arr1),
			(ctypes.c_int * len(arr2)) (*arr2),
			(ctypes.c_int * len(arr3)) (*arr3),
			(ctypes.c_int * len(arr4)) (*arr4),
			m, 
			n,
			buffering
		)
	result = []
	for i in range(m*n):
		result.append( buffering[i] )
	return result

if __name__ == '__main__':
# wrong example:
#	arr1 = [[1,2,3,3,4],[20,20],[20,20]]
#	arr2 = [[1,5,3,4],[30,30,30],[30,30,30]]
# correct example:
	arr1 = [1,2,3,3,4,20,20,20,20]
	arr2 = [1,5,3,4,30,30,30,30,30,30]
	arr3 = [5,2,2]
	arr4 = [4,3,3]
	result = get_ed_btwn_lists(arr1, arr2, arr3, arr4)
# 	result: [2,5,5,4,3,3,4,3,3]	
	print(result)	

#	print(type(buff))
#	<class '__main__.c_int_Array_9'>

#	print(buff)
#	<__main__.c_int_Array_9 object at 0x...>
#	for i in range(9):
#		print(buff[i])
