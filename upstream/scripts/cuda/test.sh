
set -xe

backend=cuda
# data="data_$(date +%F)"
data="data_$(date +%YWW%W)"
rm -rf ${data}

mkdir -p ${data}/summary
mkdir -p ${data}/cases
mkdir -p ${data}/report

bash parse.sh ${backend}_inductor ${data}
bash parse.sh ${backend} ${data}
bash parse.sh xpu/stock_xpu ${data}
bash parse.sh xpu/distributed ${data}
bash parse.sh xpu/xpu-ops ${data}

bash parse_log.sh xpu/stock_xpu ${data}


report_non_inductor_dir="${data}/report/non-inductor/"
report_inductor_dir="${data}/report/inductor/"
mkdir ${report_non_inductor_dir}
mkdir ${report_inductor_dir}

# non inductor
python merge.py --f1 ${data}/summary/${backend}.csv --f2 ${data}/summary/stock_xpu.csv --f3 ${data}/summary/xpu-ops.csv --f4 ${data}/summary/distributed.csv
mv merged.csv ${report_non_inductor_dir}/merged_NON_inductor.csv

python merge_cases.py --f1 ${data}/cases/details_${backend}.csv --f2 ${data}/cases/details_stock_xpu.csv  --f3 ${data}/cases/details_xpu-ops.csv --f4 ${data}/cases/details_distributed.csv --notmatch /inductor/
mv merged_details*.csv ${report_non_inductor_dir}
mv xpu_only_skipped.csv ${report_non_inductor_dir}
mv xpu_all_skipped.csv ${report_non_inductor_dir}

python gen_excel.py --path ${report_non_inductor_dir}
mv merged_data.xlsx ${report_non_inductor_dir}/merged_data_NON_inductor.xlsx


# inductor
python merge_inductor.py --f1 ${data}/summary/${backend}.csv --f2 ${data}/summary/stock_xpu.csv
mv merged_inductor.csv ${report_inductor_dir}/merged_inductor.csv

python merge_cases.py --f1 ${data}/cases/details_${backend}.csv --f2 ${data}/cases/details_stock_xpu.csv  --f3 ${data}/cases/details_xpu-ops.csv --f4 ${data}/cases/details_distributed.csv --match /inductor/
mv merged_details*.csv ${report_inductor_dir}
mv xpu_only_skipped.csv ${report_inductor_dir}
mv xpu_all_skipped.csv ${report_inductor_dir}

sed -i 's/|/,/g' ${data}/summary/${backend}_inductor.csv
cp ${data}/summary/${backend}_inductor.csv ${report_inductor_dir}

python gen_excel.py --path ${report_inductor_dir}
mv merged_data.xlsx ${report_inductor_dir}/merged_data_inductor.xlsx
