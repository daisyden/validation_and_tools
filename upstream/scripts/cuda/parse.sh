set -x
root=$(pwd)
backend=$1
data=$2

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

if [ -d "xml" ];then
	rm -rf xml
fi

rm *.log
rm *.csv

mkdir -p xml

find ./${backend} -name "*.xml" -print0 |while IFS= read -r -d '' xmlfile; do
    cp $xmlfile xml/.
done

if [ "$backend" = "xpu/xpu-ops" ] || [ "$backend" = "xpu/distributed" ];then
	python ./check-ut-cases.py -p ops xml/*.xml 2>&1|tee $(basename $backend).csv
else
	python ./check-ut-cases.py xml/*.xml 2>&1|tee $(basename $backend).csv
fi

mv $(basename $backend).csv ${data}/summary/.
mv details.csv ${data}/cases/details_$(basename $backend).csv

set +x
