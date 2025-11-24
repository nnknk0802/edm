#!/bin/bash
# ECG学習用Dockerヘルパースクリプト

set -e

# 色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ヘルプメッセージ
print_help() {
    cat << EOF
ECG学習用Dockerヘルパースクリプト

使用方法:
    $0 [コマンド] [オプション]

コマンド:
    build                  Dockerイメージをビルド
    test                   テストデータで動作確認
    shell                  インタラクティブシェルを起動
    create-dummy           ダミーデータセットを作成
    verify                 データセットを検証
    train                  学習を実行
    generate               心電図を生成
    logs                   学習ログを表示
    clean                  不要なDockerリソースを削除

例:
    $0 build
    $0 test
    $0 shell
    $0 create-dummy 100
    $0 train datasets/ecg_real
    $0 generate training-runs/00000-*/network-snapshot-010000.pkl

詳細: ECG_DOCKER_GUIDE.md を参照してください
EOF
}

# Dockerコマンドのベース
DOCKER_RUN="docker run --rm --gpus all -v \$(pwd):/workspace -w /workspace"
IMAGE_NAME="edm-ecg:latest"

# Dockerイメージをビルド
cmd_build() {
    echo -e "${BLUE}Dockerイメージをビルド中...${NC}"
    docker build --tag ${IMAGE_NAME} .
    echo -e "${GREEN}✓ ビルド完了${NC}"
}

# テスト実行
cmd_test() {
    echo -e "${BLUE}テストを実行中...${NC}"

    # ダミーデータ作成
    echo -e "${YELLOW}1. ダミーデータセットを作成${NC}"
    ${DOCKER_RUN} ${IMAGE_NAME} \
        python preprocess_ecg_data.py create_dummy \
            --output_dir datasets/ecg_test \
            --num_samples 100

    # データ検証
    echo -e "${YELLOW}2. データセットを検証${NC}"
    ${DOCKER_RUN} ${IMAGE_NAME} \
        python preprocess_ecg_data.py verify \
            --dataset_dir datasets/ecg_test

    # テスト学習
    echo -e "${YELLOW}3. テスト学習を実行${NC}"
    ${DOCKER_RUN} ${IMAGE_NAME} \
        python train.py \
            --outdir=training-runs \
            --data=datasets/ecg_test \
            --cond=0 \
            --arch=ddpmpp \
            --duration=1 \
            --batch=32 \
            --batch-gpu=32 \
            --tick=1 \
            --snap=5 \
            --seed=42

    echo -e "${GREEN}✓ テスト完了${NC}"
}

# インタラクティブシェル
cmd_shell() {
    echo -e "${BLUE}インタラクティブシェルを起動...${NC}"
    docker run --rm -it --gpus all \
        -v $(pwd):/workspace \
        -w /workspace \
        --shm-size=8g \
        ${IMAGE_NAME} \
        bash
}

# ダミーデータ作成
cmd_create_dummy() {
    local num_samples=${1:-100}
    echo -e "${BLUE}ダミーデータセットを作成中 (${num_samples}サンプル)...${NC}"

    mkdir -p datasets/ecg_test

    ${DOCKER_RUN} ${IMAGE_NAME} \
        python preprocess_ecg_data.py create_dummy \
            --output_dir datasets/ecg_test \
            --num_samples ${num_samples}

    echo -e "${GREEN}✓ 作成完了: datasets/ecg_test${NC}"
}

# データセット検証
cmd_verify() {
    local dataset_dir=${1:-datasets/ecg_test}
    echo -e "${BLUE}データセットを検証中: ${dataset_dir}${NC}"

    ${DOCKER_RUN} ${IMAGE_NAME} \
        python preprocess_ecg_data.py verify \
            --dataset_dir ${dataset_dir}
}

# 学習実行
cmd_train() {
    local data_dir=${1:-datasets/ecg_test}
    local duration=${2:-1}
    local batch=${3:-32}
    local gpus=${4:-1}

    echo -e "${BLUE}学習を開始...${NC}"
    echo -e "  データ: ${data_dir}"
    echo -e "  Duration: ${duration}"
    echo -e "  Batch: ${batch}"
    echo -e "  GPUs: ${gpus}"

    if [ ${gpus} -gt 1 ]; then
        # 複数GPU
        docker run --rm --gpus all \
            -v $(pwd):/workspace \
            -w /workspace \
            --shm-size=8g \
            ${IMAGE_NAME} \
            torchrun --standalone --nproc_per_node=${gpus} train.py \
                --outdir=training-runs \
                --data=${data_dir} \
                --cond=0 \
                --arch=ddpmpp \
                --duration=${duration} \
                --batch=${batch} \
                --tick=50 \
                --snap=50 \
                --seed=42
    else
        # 単一GPU
        ${DOCKER_RUN} ${IMAGE_NAME} \
            python train.py \
                --outdir=training-runs \
                --data=${data_dir} \
                --cond=0 \
                --arch=ddpmpp \
                --duration=${duration} \
                --batch=${batch} \
                --batch-gpu=${batch} \
                --tick=10 \
                --snap=50 \
                --seed=42
    fi

    echo -e "${GREEN}✓ 学習完了${NC}"
}

# 心電図生成
cmd_generate() {
    local network=${1}
    local num_samples=${2:-10}
    local steps=${3:-18}

    if [ -z "${network}" ]; then
        echo -e "${RED}エラー: モデルファイルを指定してください${NC}"
        echo "使用方法: $0 generate <model.pkl> [num_samples] [steps]"
        exit 1
    fi

    echo -e "${BLUE}心電図を生成中...${NC}"
    echo -e "  モデル: ${network}"
    echo -e "  サンプル数: ${num_samples}"
    echo -e "  ステップ数: ${steps}"

    ${DOCKER_RUN} ${IMAGE_NAME} \
        python generate_ecg.py \
            --network ${network} \
            --outdir generated_ecg \
            --num_samples ${num_samples} \
            --steps ${steps}

    echo -e "${GREEN}✓ 生成完了: generated_ecg/${NC}"
}

# ログ表示
cmd_logs() {
    local logfile=$(ls -t training-runs/*/log.txt 2>/dev/null | head -n 1)

    if [ -z "${logfile}" ]; then
        echo -e "${RED}ログファイルが見つかりません${NC}"
        exit 1
    fi

    echo -e "${BLUE}ログを表示: ${logfile}${NC}"
    tail -f ${logfile}
}

# クリーンアップ
cmd_clean() {
    echo -e "${YELLOW}不要なDockerリソースを削除中...${NC}"
    docker system prune -f
    echo -e "${GREEN}✓ クリーンアップ完了${NC}"
}

# メインロジック
case "${1}" in
    build)
        cmd_build
        ;;
    test)
        cmd_test
        ;;
    shell)
        cmd_shell
        ;;
    create-dummy)
        cmd_create_dummy ${2}
        ;;
    verify)
        cmd_verify ${2}
        ;;
    train)
        cmd_train ${2} ${3} ${4} ${5}
        ;;
    generate)
        cmd_generate ${2} ${3} ${4}
        ;;
    logs)
        cmd_logs
        ;;
    clean)
        cmd_clean
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        echo -e "${RED}エラー: 不明なコマンド '${1}'${NC}"
        echo ""
        print_help
        exit 1
        ;;
esac
