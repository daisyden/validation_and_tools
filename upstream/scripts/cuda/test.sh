if [ -d data ];then
	rm -rf data
fi

mkdir -p data/summary
mkdir -p data/cases
mkdir -p data/report

bash parse.sh cuda_inductor
bash parse.sh cuda
bash parse.sh xpu/stock_xpu
bash parse.sh xpu/xpu-ops
bash parse_log.sh xpu/stock_xpu

python python parse_dist_summary.py dist_summary.csv xpu/distributed/dist_summary_parsed.csv


cp xpu/distributed/dist_summary_parsed.csv data/summary/.
cp ../log_parser/collected.csv data/collected.csv

python merge.py --f1 data/summary/cuda.csv --f2 data/summary/stock_xpu.csv --f3 data/summary/xpu-ops.csv --f4 data/summary/dist_summary_parsed.csv
mv merged.csv data/report/merged.csv

python merge_inductor.py --f1 data/summary/cuda.csv --f2 data/summary/stock_xpu.csv
mv merged_inductor.csv data/report/merged_inductor.csv

python merge_inductor_cases.py --f1 data/cases/details_cuda.csv --f2 data/cases/details_stock_xpu.csv
mv merged_details*.csv data/report/.
mv xpu_skipped_inductor.csv data/report/.
mv xpu_skipped_inductor.txt data/report/run_skipped.csv

cp data/summary/cuda_inductor.csv data/report/.
python gen_excel.py
