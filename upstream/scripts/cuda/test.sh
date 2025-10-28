backend=$1
data="data_${backend}"
if [ -d ${data} ];then
	rm -rf ${data}
fi

mkdir -p ${data}/summary
mkdir -p ${data}/cases
mkdir -p ${data}/report

bash parse.sh ${backend}_inductor ${data}
bash parse.sh ${backend} ${data}
bash parse.sh xpu/stock_xpu ${data}
bash parse.sh xpu/distributed ${data}
bash parse.sh xpu/xpu-ops ${data}

bash parse_log.sh xpu/stock_xpu ${data}

python merge.py --f1 ${data}/summary/${backend}.csv --f2 ${data}/summary/stock_xpu.csv --f3 ${data}/summary/xpu-ops.csv --f4 ${data}/summary/distributed.csv
mv merged.csv ${data}/report/merged.csv
#
#python merge_inductor.py --f1 ${data}/summary/${backend}.csv --f2 ${data}/summary/stock_xpu.csv
#mv merged_inductor.csv ${data}/report/merged_inductor.csv

#python merge_inductor_cases.py --f1 ${data}/cases/details_${backend}.csv --f2 ${data}/cases/details_stock_xpu.csv
#mv merged_details*.csv ${data}/report/.
#mv xpu_skipped_inductor.csv ${data}/report/.
#mv xpu_all_skipped_inductor.csv ${data}/report/.
#mv xpu_skipped_inductor.txt ${data}/report/run_skipped.csv

python merge_cases.py --f1 ${data}/cases/details_${backend}.csv --f2 ${data}/cases/details_stock_xpu.csv  --f3 ${data}/cases/details_xpu-ops.csv --f4 ${data}/cases/details_distributed.csv --notmatch /inductor/
mv merged_details*.csv ${data}/report/.
mv xpu_skipped.csv ${data}/report/.
mv xpu_all_skipped.csv ${data}/report/.

#mv xpu_skipped.txt ${data}/report/run_skipped.csv


cp ${data}/summary/${backend}_inductor.csv ${data}/report/.
sed -i 's/|/,/g' ${data}/report/${backend}_inductor.csv
python gen_excel.py --path ${data}/report
