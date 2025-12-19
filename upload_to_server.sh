#!/bin/bash
# upload_to_server.sh
# 서버에 필요한 파일만 업로드하는 스크립트

SERVER="dok99@h2p.crc.pitt.edu"
REMOTE_DIR="~/firstpaper"

echo "=========================================="
echo "Uploading files to server"
echo "=========================================="
echo "Server: $SERVER"
echo "Remote directory: $REMOTE_DIR"
echo ""

# 1. Python 실행 파일들 업로드
echo "Step 1: Uploading Python files..."
scp DH_main.py \
    DH_config.py \
    DH_data_gen.py \
    DH_algo.py \
    DH_master.py \
    DH_sub.py \
    requirements.txt \
    $SERVER:$REMOTE_DIR/

if [ $? -eq 0 ]; then
    echo "✅ Python files uploaded successfully"
else
    echo "❌ Error uploading Python files"
    exit 1
fi

# 2. 서버 실행 스크립트들 업로드
echo ""
echo "Step 2: Uploading server scripts..."
scp setup_server.sh \
    run_server.sh \
    run_background.sh \
    submit_slurm.sh \
    generate_multiple_datasets.py \
    $SERVER:$REMOTE_DIR/

if [ $? -eq 0 ]; then
    echo "✅ Server scripts uploaded successfully"
else
    echo "❌ Error uploading server scripts"
    exit 1
fi

# 3. 서버에 data 디렉토리 생성
echo ""
echo "Step 3: Creating data directory on server..."
ssh $SERVER "mkdir -p $REMOTE_DIR/data"

if [ $? -eq 0 ]; then
    echo "✅ Data directory created"
else
    echo "❌ Error creating data directory"
    exit 1
fi

# 4. 데이터 파일들 업로드 (50개 파일)
echo ""
echo "Step 4: Uploading data files (50 datasets)..."
echo "This may take a few minutes..."
scp data/DH_data_full_seed*.pkl $SERVER:$REMOTE_DIR/data/

if [ $? -eq 0 ]; then
    echo "✅ Data files uploaded successfully"
else
    echo "❌ Error uploading data files"
    exit 1
fi

# 5. 서버에 result 디렉토리 생성
echo ""
echo "Step 5: Creating result directory on server..."
ssh $SERVER "mkdir -p $REMOTE_DIR/result"

if [ $? -eq 0 ]; then
    echo "✅ Result directory created"
else
    echo "❌ Error creating result directory"
    exit 1
fi

# 6. 스크립트 실행 권한 부여
echo ""
echo "Step 6: Setting execute permissions on server..."
ssh $SERVER "cd $REMOTE_DIR && chmod +x *.sh"

if [ $? -eq 0 ]; then
    echo "✅ Execute permissions set"
else
    echo "❌ Error setting permissions"
    exit 1
fi

# 7. 업로드 확인
echo ""
echo "Step 7: Verifying uploaded files..."
echo ""
echo "Python files:"
ssh $SERVER "ls -lh $REMOTE_DIR/*.py"
echo ""
echo "Data files:"
ssh $SERVER "ls $REMOTE_DIR/data/*.pkl | wc -l"
echo ""

echo "=========================================="
echo "✅ Upload completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. SSH to server: ssh $SERVER"
echo "2. Go to directory: cd firstpaper"
echo "3. Setup environment: bash setup_server.sh"
echo "4. Test run: python3 DH_main.py toy 5 --seed 1 --di HD"
echo "5. Full run: bash run_background.sh"
echo ""
