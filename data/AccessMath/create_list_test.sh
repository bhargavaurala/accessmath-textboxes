#!/bin/bash
#training list: lecture_01 lecture_06 lecture_18 NM_lecture_01 NM_lecture_03
#testing list: lecture_02 lecture_07 lecture_08 lecture_10 lecture_15 lecture_18 NM_lecture_02 NM_lecture_05

root_dir=$AM_DATA_DIR
sub_dir=ImageSets/Main
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for dataset in test
do
  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  for name in lecture_02 lecture_07 lecture_08 lecture_10 lecture_15 NM_lecture_02 NM_lecture_05
  do
    if [[ $dataset == "test" && $name == "VOC2012" ]]
    then
      continue
    fi
    echo "Create list for $name $dataset..."
    dataset_file=$root_dir/$name/$sub_dir/$dataset.txt

    img_file=$bash_dir/$dataset"_img.txt"
    cp $dataset_file $img_file
    sed -i "s/^/$name\/JPEGImages\//g" $img_file
    sed -i "s/$/.jpg/g" $img_file
    
    # test
    paste $img_file >> $dst_file    
    rm -f $img_file
    
  done

  # Generate image name and size infomation.
  # if [[ $dataset == "train" || $dataset == "val" ]]
  # then
  $bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  # fi

  # Shuffle trainval file.
  # if [[ $dataset == "train" || $dataset == "val" ]]
  # then
  rand_file=$dst_file.random
  cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
  mv $rand_file $dst_file
  # fi
done
