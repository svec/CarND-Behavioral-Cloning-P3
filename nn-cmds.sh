# aws-0
#./nn.py --model_type nvidia --data_dirs r-0 --model_file_output model.h5.r-0 >& nn.out.r-0
#./nn.py --model_type nvidia --data_dirs r-0,r-swerve-0 --model_file_output model.h5.r-0-s-0 >& nn.out.r-0-s-0
# aws-1
#./nn.py --model_type nvidia --data_dirs r-0,r-rev-0 --model_file_output model.h5.r-0-r-0 >& nn.out.r-0-r-0
#good!#./nn.py --model_type nvidia --data_dirs r-0,r-rev-0,r-swerve-0 --model_file_output model.h5.r-0-r-0-s-0 >& nn.out.r-0-r-0-s-0
# aws-
# Is r-2 bad???
#./nn.py --model_type nvidia --data_dirs r-2 --model_file_output model.h5.r-2 >& nn.out.r-2
#./nn.py --model_type nvidia --data_dirs r-0,r-2 --model_file_output model.h5.r-02 >& nn.out.r-02
#./nn.py --model_type nvidia --data_dirs r-0,r-2,r-rev-0,r-swerve-0 --model_file_output model.h5.r-02-r-0-s-0 >& nn.out.r-02-r-0-s-0
#./nn.py --model_type nvidia --data_dirs r-0,r-2,r-rev-0,r-swerve-0,r-swerve-1 --model_file_output model.h5.r-02-r-0-s-01 >& nn.out.r-02-r-0-s-01
#./nn.py --model_type nvidia --learning_rate 0.0001 --data_dirs r-2 --model_file_output model.h5.r-2 >& nn.out.r-2-lr.0001

# aws-2
#./nn.py --model_type nvidia --data_dirs r-0,r-2,r-rev-0,r-swerve-0,r-swerve-1,r-swerve-2,r-swerve-3,r-swerve-4,r-curve-0 --model_file_output model.h5.r-02-r-0-s-01234-c0 >& nn.out.r-02-r-0-s-01234-c0
#./nn.py --model_type nvidia --data_dirs r-0,r-2,r-rev-0,r-swerve-0,r-swerve-1,r-swerve-2,r-swerve-3,r-swerve-4,r-curve-0,r-curve-1 --model_file_output model.h5.r-02-r-0-s-01234-c01 >& nn.out.r-02-r-0-s-01234-c01 # good!
#./nn.py --model_type nvidia --data_dirs r-0,r-2,r-3,r-rev-0,r-swerve-0,r-swerve-1,r-swerve-2,r-swerve-3,r-swerve-4,r-curve-0,r-curve-1 --model_file_output model.h5.r-023-r-0-s-01234-c01 >& nn.out.r-023-r-0-s-01234-c01
#./nn.py --model_type nvidia --data_dirs data,r-0,r-2,r-rev-0,r-swerve-0,r-swerve-1,r-swerve-2,r-swerve-3,r-swerve-4,r-curve-0,r-curve-1 --model_file_output model.h5.d-r-02-r-0-s-01234-c01 >& nn.out.d-r-02-r-0-s-01234-c01 # Not good, crashes after bridge
#./nn.py --model_type nvidia --data_dirs r-0,r-2,r-3,r-rev-0,r-swerve-0,r-swerve-1,r-swerve-2,r-swerve-3,r-swerve-4,r-curve-0,r-curve-1,r-curve-2 --model_file_output model.h5.r-023-r-0-s-01234-c012 >& nn.out.r-023-r-0-s-01234-c012 # good!!! Gets to the penultimate curve
#./nn.py --model_type nvidia --data_dirs r-0,r-2,r-3,r-rev-0,r-swerve-0,r-swerve-1,r-swerve-2,r-swerve-3,r-swerve-4,r-curve-0,r-curve-1,r-curve-2,r-curve-3 --model_file_output model.h5.r-023-r-0-s-01234-c0123 >& nn.out.r-023-r-0-s-01234-c0123 # rough until the bridge, then bad
#./nn.py --model_type nvidia --data_dirs r-0,r-2,r-3,r-4,r-rev-0,r-swerve-0,r-swerve-1,r-swerve-2,r-swerve-3,r-swerve-4,r-curve-0,r-curve-1,r-curve-2,r-curve-3 --model_file_output model.h5.r-0234-r-0-s-01234-c0123 >& nn.out.r-0234-r-0-s-01234-c0123
#./nn.py --model_type nvidia --data_dirs r-0,r-2,r-3,r-4,r-5,r-rev-0,r-swerve-0,r-swerve-1,r-swerve-2,r-swerve-3,r-swerve-4,r-curve-0,r-curve-1,r-curve-2,r-curve-3 --model_file_output model.h5.r-02345-r-0-s-01234-c0123 >& nn.out.r-02345-r-0-s-01234-c0123
./nn.py --model_type nvidia --data_dirs r-0,r-2,r-3,r-4,r-rev-0,r-swerve-0,r-swerve-1,r-swerve-2,r-swerve-3,r-swerve-4,r-curve-0,r-curve-1,r-curve-2,r-curve-3,r-curve-4 --model_file_output model.h5.r-0234-r-0-s-01234-c01234 >& nn.out.r-0234-r-0-s-01234-c01234
./nn.py --model_type nvidia --data_dirs r-0,r-2,r-3,r-4,r-5,r-rev-0,r-swerve-0,r-swerve-1,r-swerve-2,r-swerve-3,r-swerve-4,r-curve-0,r-curve-1,r-curve-2,r-curve-3,r-curve-4 --model_file_output model.h5.r-02345-r-0-s-01234-c01234 >& nn.out.r-02345-r-0-s-01234-c01234

#bad#./nn.py --model_type nvidia --data_dirs r-1 --model_file_output model.h5.r-1 >& nn.out.r-1
#bad#./nn.py --model_type nvidia --data_dirs r-0,r-1 --model_file_output model.h5.r-01 >& nn.out.r-01
#bad#./nn.py --model_type nvidia --data_dirs r-0,r-1,r-swerve-0 --model_file_output model.h5.r-01-s-0 >& nn.out.r-01-s-0
#bad#./nn.py --model_type nvidia --data_dirs r-1,r-rev-0 --model_file_output model.h5.r-1-r-0 >& nn.out.r-1-r-0
#bad#./nn.py --model_type nvidia --data_dirs r-0,r-1,r-rev-0,r-swerve-0 --model_file_output model.h5.r-01-r-0-s-0 >& nn.out.r-01-r-0-s-0

#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 >& exp-nogen-0
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 >& exp-nogen-1
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 >& exp-nogen-2
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 >& exp-nogen-3
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 >& exp-nogen-4
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 >& exp-nogen-5
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 >& exp-nogen-6
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 >& exp-nogen-7
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 >& exp-nogen-8
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 >& exp-nogen-9

#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.002 >& exp-nogen-lr002-0
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.002 >& exp-nogen-lr002-1
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.002 >& exp-nogen-lr002-2
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.002 >& exp-nogen-lr002-3
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.002 >& exp-nogen-lr002-4
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.002 >& exp-nogen-lr002-5
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.002 >& exp-nogen-lr002-6
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.002 >& exp-nogen-lr002-7
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.002 >& exp-nogen-lr002-8
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.002 >& exp-nogen-lr002-9

#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.0001 >& exp-nogen-lr0001-0
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.0001 >& exp-nogen-lr0001-1
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.0001 >& exp-nogen-lr0001-2
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.0001 >& exp-nogen-lr0001-3
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.0001 >& exp-nogen-lr0001-4
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.0001 >& exp-nogen-lr0001-5
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.0001 >& exp-nogen-lr0001-6
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.0001 >& exp-nogen-lr0001-7
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.0001 >& exp-nogen-lr0001-8
#time ./nogen-nn.py --model_type nvidia --data_dirs r-0 --learning_rate 0.0001 >& exp-nogen-lr0001-9

#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 32 >& exp-32-0
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 32 >& exp-32-1
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 32 >& exp-32-2
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 32 >& exp-32-3
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 32 >& exp-32-4
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 32 >& exp-32-5
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 32 >& exp-32-6
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 32 >& exp-32-7
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 32 >& exp-32-8
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 32 >& exp-32-9

#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 10 >& exp-10-0
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 10 >& exp-10-1
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 10 >& exp-10-2
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 10 >& exp-10-3
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 10 >& exp-10-4
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 10 >& exp-10-5
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 10 >& exp-10-6
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 10 >& exp-10-7
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 10 >& exp-10-8
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 10 >& exp-10-9

#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 30 >& exp-30-0
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 30 >& exp-30-1
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 30 >& exp-30-2
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 30 >& exp-30-3
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 30 >& exp-30-4
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 30 >& exp-30-5
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 30 >& exp-30-6
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 30 >& exp-30-7
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 30 >& exp-30-8
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 30 >& exp-30-9

#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 64 >& exp-64-0
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 64 >& exp-64-1
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 64 >& exp-64-2
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 64 >& exp-64-3
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 64 >& exp-64-4
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 64 >& exp-64-5
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 64 >& exp-64-6
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 64 >& exp-64-7
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 64 >& exp-64-8
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 64 >& exp-64-9

#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 100 >& exp-100-0
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 100 >& exp-100-1
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 100 >& exp-100-2
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 100 >& exp-100-3
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 100 >& exp-100-4
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 100 >& exp-100-5
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 100 >& exp-100-6
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 100 >& exp-100-7
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 100 >& exp-100-8
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 100 >& exp-100-9

#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 128 >& exp-128-0
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 128 >& exp-128-1
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 128 >& exp-128-2
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 128 >& exp-128-3
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 128 >& exp-128-4
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 128 >& exp-128-5
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 128 >& exp-128-6
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 128 >& exp-128-7
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 128 >& exp-128-8
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 128 >& exp-128-9

#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 256 >& exp-256-0
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 256 >& exp-256-1
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 256 >& exp-256-2
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 256 >& exp-256-3
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 256 >& exp-256-4
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 256 >& exp-256-5
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 256 >& exp-256-6
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 256 >& exp-256-7
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 256 >& exp-256-8
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 256 >& exp-256-9

#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 512 >& exp-512-0
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 512 >& exp-512-1
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 512 >& exp-512-2
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 512 >& exp-512-3
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 512 >& exp-512-4
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 512 >& exp-512-5
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 512 >& exp-512-6
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 512 >& exp-512-7
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 512 >& exp-512-8
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 512 >& exp-512-9

#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1024 >& exp-1024-0
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1024 >& exp-1024-1
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1024 >& exp-1024-2
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1024 >& exp-1024-3
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1024 >& exp-1024-4
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1024 >& exp-1024-5
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1024 >& exp-1024-6
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1024 >& exp-1024-7
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1024 >& exp-1024-8
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1024 >& exp-1024-9

#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1000 >& exp-1000-0
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1000 >& exp-1000-1
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1000 >& exp-1000-2
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1000 >& exp-1000-3
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1000 >& exp-1000-4
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1000 >& exp-1000-5
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1000 >& exp-1000-6
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1000 >& exp-1000-7
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1000 >& exp-1000-8
#time ./nn.py --model_type nvidia --data_dirs r-0 --batch_size 1000 >& exp-1000-9
