set -x
root=$(pwd)
backend=$1
find ./${backend} -name "*.zip" -print0 | while IFS= read -r -d '' zipfile; do
    if [[ "$zipfile" =~ distributed_weekly ]];then
           continue
    fi 
    if [[ "$zipfile" =~ torch-xpu-ops ]];then
           continue
    fi 

    echo $zipfile
    echo "Processing: $zipfile"

    dirname="${zipfile%.zip}"
    basename=$(basename $zipfile .zip)
    workflow=$(echo "$dirname" | awk -F'/' '{print $(NF-1)}')
    if [ -d $dirname ];then
            rm -rf $dirname
    fi
    mkdir -p "$dirname"
    unzip -o -q "$zipfile" -d "$dirname"
done

if [ -d "logs" ];then
	rm -rf logs
fi

rm *.log
rm *.csv

mkdir -p logs 

find ./${backend} -name "*.log" -print0 |while IFS= read -r -d '' xmlfile; do
    cp $xmlfile logs/.
done

python collected_cases.py logs/*.log 2>&1|tee data/collected.csv

set +x
