set -x
root=$(pwd)
backend=$1

if [ -d "logs" ];then
	rm -rf logs
fi

rm -f *.log
rm -f *.csv

mkdir -p logs 

find ./${backend} -name "*.log" -print0 |while IFS= read -r -d '' xmlfile; do
    cp $xmlfile logs/.
done

python collected_cases.py logs/*.log 2>&1|tee data/collected.csv

set +x
