downloadDataset:
	python3 src/downloadDataset.py && mv dataset/DASPS_Database/* dataset/ 

testDataset:
	python3 src/dataset.py