# aws-0
#./nn.py --model_type nvidia --data_dirs r-0 --model_file_output model.h5.r-0 >& nn.out.r-0
#./nn.py --model_type nvidia --data_dirs r-0,r-swerve-0 --model_file_output model.h5.r-0-s-0 >& nn.out.r-0-s-0
# aws-1
#./nn.py --model_type nvidia --data_dirs r-0,r-rev-0 --model_file_output model.h5.r-0-r-0 >& nn.out.r-0-r-0
#good!#./nn.py --model_type nvidia --data_dirs r-0,r-rev-0,r-swerve-0 --model_file_output model.h5.r-0-r-0-s-0 >& nn.out.r-0-r-0-s-0
# aws-
#./nn.py --model_type nvidia --data_dirs r-2 --model_file_output model.h5.r-2 >& nn.out.r-2
./nn.py --model_type nvidia --data_dirs r-0,r-2 --model_file_output model.h5.r-02 >& nn.out.r-02
./nn.py --model_type nvidia --data_dirs r-0,r-2,r-rev-0,r-swerve-0 --model_file_output model.h5.r-02-r-0-s-0 >& nn.out.r-02-r-0-s-0
./nn.py --model_type nvidia --data_dirs r-0,r-2,r-rev-0,r-swerve-0,r-swerve-1 --model_file_output model.h5.r-02-r-0-s-01 >& nn.out.r-02-r-0-s-01

#bad#./nn.py --model_type nvidia --data_dirs r-1 --model_file_output model.h5.r-1 >& nn.out.r-1
#bad#./nn.py --model_type nvidia --data_dirs r-0,r-1 --model_file_output model.h5.r-01 >& nn.out.r-01
#bad#./nn.py --model_type nvidia --data_dirs r-0,r-1,r-swerve-0 --model_file_output model.h5.r-01-s-0 >& nn.out.r-01-s-0
#bad#./nn.py --model_type nvidia --data_dirs r-1,r-rev-0 --model_file_output model.h5.r-1-r-0 >& nn.out.r-1-r-0
#bad#./nn.py --model_type nvidia --data_dirs r-0,r-1,r-rev-0,r-swerve-0 --model_file_output model.h5.r-01-r-0-s-0 >& nn.out.r-01-r-0-s-0
