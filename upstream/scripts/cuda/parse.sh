set -x
root=$(pwd)
backend=$1
data=$2


if [ -d "xml" ];then
	rm -rf xml
fi

rm -f *.log
rm -f *.csv

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
