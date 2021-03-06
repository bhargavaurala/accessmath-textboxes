import os
import sys
from random import shuffle

trainval_list_file = 'trainval.txt'
train_list_file = 'train.txt'
val_list_file = 'val.txt'
test_list_file = 'test.txt'

training_lectures = ['lecture_01', 'lecture_06', 'lecture_18', 'NM_lecture_01', 'NM_lecture_03']
testing_lectures = ['lecture_02', 'lecture_07', 'lecture_08', 'lecture_10', 'lecture_15', 'NM_lecture_02', 'NM_lecture_05']
lectures_set = training_lectures
# lectures_set = testing_lectures
is_test = lectures_set == testing_lectures

def split(lines, fraction=0.8):
	tr = int(len(lines) * fraction)
	return lines[: tr], lines[tr :]

if __name__ == "__main__":
	"""
	Run using python create_list.py $AM_DATA_DIR
	"""
	img_path = sys.argv[1]
	for lecture in lectures_set:
		trainval = []
		test = []
		xml_path = os.path.join(img_path, lecture, 'Annotations')
		_, _, files = os.walk(os.path.join(img_path, lecture, 'JPEGImages')).next()
		img_files = [f for f in files if '.jpg' in f]	
		for img_file in img_files:
			img_id = img_file.split('.')[0]
			# check if corresponding xml exists
			xml_file = os.path.join(xml_path, img_id + '.xml')
			if not os.path.isfile(xml_file) and not is_test:
				continue
			if is_test:
				test += [img_id]
			else:		
				trainval += [img_id]
		if len(trainval) > 0:
			shuffle(trainval)
			print lecture, len(trainval)
			train, val = split(trainval)
			with open(os.path.join(img_path, lecture, 'ImageSets', 'Main', trainval_list_file), 'w') as f:
				f.write('\n'.join(trainval))
			with open(os.path.join(img_path, lecture, 'ImageSets', 'Main', train_list_file), 'w') as f:
				f.write('\n'.join(train))
			with open(os.path.join(img_path, lecture, 'ImageSets', 'Main', val_list_file), 'w') as f:	
				f.write('\n'.join(val))
		elif len(test) > 0:
			shuffle(test)
			print lecture, len(test)
			with open(os.path.join(img_path, lecture, 'ImageSets', 'Main', test_list_file), 'w') as f:	
				f.write('\n'.join(test))
		else:
			pass


