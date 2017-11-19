import os 
import glob
import time

max_model = 2
prefix = 'model.ckpt-'
print('Monitor to del model.')
while True:
	index_files = glob.glob("*.index")
	models = [ int(index.replace(prefix, '').split('.')[0]) for index in index_files]

	#print(models)
	models.sort()
	try:
		if len(models) > max_model:
			for name in glob.glob(prefix + str(models[0]) + "*"):
				os.remove(name)
				print("Deleted: ", name)
	except Exception as e:
		pinrt(e)
		print('error occured.')
	print("sleep 30s")
	time.sleep(30)

