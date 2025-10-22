#if [ -d data ];then
#	rm -rf data
#fi
#
#mkdir -p data/summary
#mkdir -p data/cases
#mkdir -p data/report
#
#bash parse.sh cuda_inductor
#bash parse.sh cuda
#bash parse.sh xpu/stock_xpu
#bash parse.sh xpu/xpu-ops
#
#cp xpu/distributed/dist_summary_parsed.csv data/summary/.

#python merge.py --f1 data/summary/cuda.csv --f2 data/summary/stock_xpu.csv --f3 data/summary/xpu-ops.csv --f4 data/summary/dist_summary_parsed.csv
#mv merged.csv data/report/merged.csv

#python merge_inductor.py --f1 data/summary/cuda.csv --f2 data/summary/stock_xpu.csv
#mv merged_inductor.csv data/report/merged_inductor.csv
#
python merge_inductor_cases.py --f1 data/cases/details_cuda.csv --f2 data/cases/details_stock_xpu.csv
mv merged_details*.csv data/report/.
