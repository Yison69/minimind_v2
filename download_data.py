from modelscope.hub.snapshot_download import snapshot_download
import os

DATASET_ID = "gongjy/minimind_dataset"  # 你的数据集ID
SAVE_DIR = "./dataset"

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"开始下载数据集 {DATASET_ID} 到 {SAVE_DIR}...")
try:
    snapshot_download(
        repo_id=DATASET_ID,
        local_dir=SAVE_DIR,
        repo_type="dataset",  # 明确是数据集（旧版也支持）
        #file_pattern="sft_mini_512.jsonl"  # 替换成你要下载的文件名（旧版参数名）
    )
    print(f"✅ 数据集下载完成！文件保存在：{os.path.abspath(SAVE_DIR)}")
except Exception as e:
    print(f"❌ 下载失败：{str(e)}")
    print("\n解决方法：")
    print("1. 确认文件名正确（比如你要下的是 sft_mini_512.jsonl，和数据集文件列表一致）；")
    print("2. 若数据集私有，先执行 `python -m modelscope login` 登录；")
    print("3. 若要下载全部文件，删除 `file_regex` 这一行即可。")