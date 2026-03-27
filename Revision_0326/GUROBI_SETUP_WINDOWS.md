# Windows에서 Gurobi 설치 및 라이선스 설정

> 전제: 학교 VPN (GlobalProtect, portal-palo.pitt.edu)에 이미 연결된 상태

---

## Step 1: Gurobi 계정 생성

1. https://portal.gurobi.com/iam/register/ 접속
2. **학교 이메일** (pitt.edu)로 가입
3. 이메일 인증 완료

---

## Step 2: Academic License 발급

1. 로그인 후 https://www.gurobi.com/downloads/end-user-license-agreement-academic/ 접속
2. Academic License 약관 동의
3. **License key**가 표시됨 (형식: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)
4. 이 키를 복사해둠

---

## Step 3: Gurobi 설치

### 방법 A: pip으로 설치 (간단)
```
pip install gurobipy
```

### 방법 B: 공식 설치파일 (pip이 안 되면)
1. https://www.gurobi.com/downloads/ 에서 Windows 64-bit 다운로드
2. 설치파일 실행 (기본 경로: `C:\gurobi1100\` 등)
3. 설치 후 pip으로도 추가: `pip install gurobipy`

---

## Step 4: 라이선스 활성화 (VPN 연결 상태에서!)

**명령 프롬프트(cmd)** 또는 **PowerShell** 열고:

```
grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

(xxxxxxxx... 부분에 Step 2에서 복사한 키를 붙여넣기)

실행하면:
```
info  : grbgetkey version 11.x.x
info  : Contacting Gurobi key server...
info  : Key for license ID xxxxxx was successfully retrieved
info  : License expires at 20xx-xx-xx
info  : Saving license key...

In which directory would you like to store the Gurobi license key file?
[hit Enter to store it in C:\Users\<username>]:
```

**Enter를 누르면 기본 경로에 저장됨** (`C:\Users\<username>\gurobi.lic`)

---

## Step 5: 설치 확인

```
python -c "import gurobipy; m = gurobipy.Model(); print('Gurobi OK')"
```

`Gurobi OK`가 출력되면 성공. 이제 **VPN을 꺼도** Gurobi가 정상 작동합니다.

---

## 문제 해결

| 증상 | 해결 |
|------|------|
| `grbgetkey`를 찾을 수 없음 | Gurobi 설치 경로를 PATH에 추가: `set PATH=%PATH%;C:\gurobi1100\win64\bin` |
| `HostID mismatch` | VPN이 연결된 상태에서 다시 시도 |
| `License expired` | https://portal.gurobi.com/ 에서 새 키 발급 후 다시 `grbgetkey` |
| `import gurobipy` 실패 | `pip install gurobipy` 재실행 |
| `No Gurobi license found` | `gurobi.lic` 파일 위치 확인: `echo %GUROBI_HOME%` 또는 `C:\Users\<username>\gurobi.lic` 존재하는지 확인 |
