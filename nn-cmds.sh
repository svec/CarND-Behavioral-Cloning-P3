./nn.py --model_type nvidia --data_dirs r-0 --model_file_output model.h5.r-0 >& nn.out.r-0
./nn.py --model_type nvidia --data_dirs r-1 --model_file_output model.h5.r-1 >& nn.out.r-1
./nn.py --model_type nvidia --data_dirs r-0,r-1 --model_file_output model.h5.r-01 >& nn.out.r-01
./nn.py --model_type nvidia --data_dirs r-0,r-swerve-0 --model_file_output model.h5.r-0-s-0 >& nn.out.r-0-s-0
./nn.py --model_type nvidia --data_dirs r-0,r-1,r-swerve-0 --model_file_output model.h5.r-01-s-0 >& nn.out.r-01-s-0
./nn.py --model_type nvidia --data_dirs r-1,r-rev-0 --model_file_output model.h5.r-1-r-0 >& nn.out.r-1-r-0
./nn.py --model_type nvidia --data_dirs r-0,r-1,r-rev-0,r-swerve-0 --model_file_output model.h5.r-01-r-0-s-0 >& nn.out.r-01-r-0-s-0
