#!/usr/bin/env bash
# 换租用服务器时一键配置（恒源云 gpushare；镜像自带 Python 3.8 + torch 1.9.1+cu111）。用法见 CLAUDE.md 第 8 节"换服务器"。
#
# 前提：已把本地的 Res/evuav_migrate.tar.gz 上传到 /hy-tmp/（含 datasets/EV-UAV-dataset、V5 权重目录、训练统计文件）。
# 用法：bash setup_rented_server.sh [分支名，默认 v6-4]
#   不用 git、代码用 SFTP 上传时：SKIP_GIT=1 bash /hy-tmp/EV-UAV-v1/tools/setup_rented_server.sh
#   第一次在新机器上运行时，本脚本还不在机器上：先执行 CLAUDE.md 第 8 节里的 git clone 一行，再运行
#   bash /hy-tmp/EV-UAV-v1/tools/setup_rented_server.sh
# 每一步都可以重复执行（已完成的步骤会跳过）；最后打印每项检查的 PASS / FAIL。
set -u
BRANCH="${1:-v6-4}"
REPO="https://github.com/wangyong334/baseline.git"
ROOT=/hy-tmp
CODE=$ROOT/EV-UAV-v1
ARCHIVE=$ROOT/evuav_migrate.tar.gz
LEGACY=/media/stephen/nvme0n1/wy_data/datasets
EXPECT_TESTS="${EXPECT_TESTS:-}"          # 例：EXPECT_TESTS=216 bash setup_rented_server.sh；为空时只要求 OK
declare -a REPORT

ok()   { REPORT+=("PASS  $1"); echo "[PASS] $1"; }
bad()  { REPORT+=("FAIL  $1"); echo "[FAIL] $1"; }

echo "== 1. 代码（$BRANCH）"
if [ "${SKIP_GIT:-0}" = "1" ]; then
  [ -f "$CODE/train_stream_v1.py" ] && ok "跳过 git（代码已用 SFTP 上传）" || bad "SKIP_GIT=1 但 $CODE 里没有代码：先用 SFTP 上传项目"
elif [ -d "$CODE/.git" ]; then
  (cd "$CODE" && git fetch -q origin && git checkout -q "$BRANCH" && git pull -q origin "$BRANCH") \
    && ok "代码已更新到 $(cd "$CODE" && git log --oneline -1)" || bad "git pull 失败（私有仓库需要登录，或网络不通；可改用本地 git archive 上传）"
else
  git clone -q -b "$BRANCH" --single-branch "$REPO" "$CODE" \
    && ok "代码已克隆：$(cd "$CODE" && git log --oneline -1)" || bad "git clone 失败"
fi
mkdir -p "$CODE/run" "$CODE/log"

echo "== 2. 数据、V5 权重、训练统计（解压 $ARCHIVE 到 $ROOT）"
if [ -f "$CODE/log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt" ] && [ -d "$ROOT/datasets/EV-UAV-dataset/test" ]; then
  ok "已解压过，跳过"
elif [ -f "$ARCHIVE" ]; then
  tar -xzf "$ARCHIVE" -C "$ROOT" && ok "解压完成" || bad "解压失败（压缩包可能没传完整，核对大小：本地 387658040 字节）"
else
  bad "找不到 $ARCHIVE：先把本地 Res/evuav_migrate.tar.gz 上传到 $ROOT/"
fi

echo "== 3. 旧数据路径的软链接（配置文件不用改）"
mkdir -p "$LEGACY"
if [ -e "$LEGACY/EV-UAV-dataset" ]; then ok "软链接已存在"; else
  ln -s "$ROOT/datasets/EV-UAV-dataset" "$LEGACY/EV-UAV-dataset" && ok "软链接已建立" || bad "软链接失败"; fi

echo "== 4. 依赖（系统 Python；不用 conda）"
python -m pip install -q pyyaml tqdm "numpy<1.25" pandas -i https://pypi.tuna.tsinghua.edu.cn/simple \
  && ok "pip 依赖" || bad "pip 安装失败"
if ! python -c "import cv2" 2>/dev/null; then
  (apt-get update -qq && apt-get install -y -qq libglib2.0-0 libgl1) >/dev/null 2>&1 || true
  python -c "import cv2" 2>/dev/null || python -m pip install -q opencv-python-headless==4.8.1.78 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
fi
python -c "import numpy,yaml,pandas,tqdm,cv2,torch;print('numpy',numpy.__version__,'pandas',pandas.__version__,'cv2',cv2.__version__,'torch',torch.__version__)" \
  && ok "Python 包齐全" || bad "Python 包缺失"

echo "== 5. GPU 与 torch"
nvidia-smi -L
python -c "import torch;assert torch.cuda.is_available();x=torch.randn(2048,2048,device='cuda');print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),'%.1f GiB'%(torch.cuda.get_device_properties(0).total_memory/2**30),float((x@x).sum()))" \
  && ok "GPU 可用" || bad "GPU 不可用"

echo "== 6. 数据与权重"
for s in train val test; do echo "  $s: $(ls "$ROOT/datasets/EV-UAV-dataset/$s"/*.npz 2>/dev/null | wc -l) 个 npz"; done
n=$(ls "$ROOT"/datasets/EV-UAV-dataset/{train,val,test}/*.npz 2>/dev/null | wc -l)
[ "$n" -eq 147 ] && ok "数据 147 段（99/24/24）" || bad "数据段数 $n（应为 147）"
(cd "$CODE" && python -c "import torch;c=torch.load('log/stream_tif_commpeak_msg_seed37/best_val_iou_seed37.pt',map_location='cpu');print('V5 epoch',c['epoch'])") \
  && ok "V5 权重可读" || bad "V5 权重不可读"
[ -f "$CODE/configs/stream_v1_train_stats.json" ] && ok "训练统计文件存在" || bad "缺 configs/stream_v1_train_stats.json"

echo "== 7. 单元测试"
out=$(cd "$CODE" && python -m unittest discover -s tests -p "test_stream*.py" 2>&1 | tail -3)
echo "$out"
if echo "$out" | grep -q "^OK"; then
  ran=$(echo "$out" | grep -oE "Ran [0-9]+" | grep -oE "[0-9]+")
  if [ -n "$EXPECT_TESTS" ] && [ "$ran" != "$EXPECT_TESTS" ]; then bad "测试 $ran 项（应为 $EXPECT_TESTS）"; else ok "测试 $ran 项 OK"; fi
else bad "单元测试未通过"; fi

echo; echo "================ 汇总 ================"
printf '%s\n' "${REPORT[@]}"
echo "命令开头写：cd $CODE && which python（应为 /usr/bin/python；不要 conda activate）"
