set -x
runid=$1
# if [ -e "./nightly/artifacts/${runid}" ]; then
#     echo "Artifacts for runid ${runid} already downloaded."
#     exit 0
# fi
# mkdir -p ./nightly/artifacts/${runid}
# pushd ./nightly/artifacts/${runid}
# GITHUB_TOKEN=${GITHUB_TOKEN} gh --repo intel/torch-xpu-ops run download $runid
# popd
xml_dir=$(ls ./nightly/artifacts/${runid}/Inductor*-op_ut-*/ | head -n 1)
xml_dir=./nightly/artifacts/${runid}/Inductor*-op_ut-*/${xml_dir}
python check-ut.py -i ./${xml_dir}/*.xml

if [ ! -e "dynamic_skip_list_xpuindex.xlsx" ]; then
    echo "No dynamic_skip_list_xpuindex.xlsx found, only outputting raw information."
fi

GITHUB_TOKEN=${GITHUB_TOKEN} python collect_skipped_issue_info.py --artifacts ${xml_dir}

set +x