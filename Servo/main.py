import serial

# --- 配置区 ---
# 请根据你的实际情况修改端口号（Windows通常是COM3等，Mac/Linux通常是/dev/ttyUSB0）
PORT = 'COM3' 
BAUDRATE = 115200

try:
    # 打开串口
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"✅ 成功连接串口: {PORT} (波特率: {BAUDRATE})")
    print("👉 提示: 请输入4个浮点数(范围 -1 到 1)，用空格分隔。输入 'q' 退出。")
    print("👉 例如输入: 0.5 -0.5 1.0 0.0\n")

    while True:
        # 获取用户输入
        user_input = input("请输入4个参数: ")
        
        if user_input.lower() == 'q':
            break
            
        try:
            # 提取输入的数字
            params = user_input.split()
            if len(params) != 4:
                print("❌ 错误：必须输入4个数字！\n")
                continue
                
            # 转换为浮点数
            f1, f2, f3, f4 = map(float, params)
            
            # 拼接成 STM32 需要的格式（末尾自带 \n）
            send_str = f"{f1},{f2},{f3},{f4}\n"
            
            # 编码并发送
            ser.write(send_str.encode('utf-8'))
            print(f"📤 已发送: {send_str.strip()}\n")
            
        except ValueError:
            print("❌ 错误：包含无效的数字，请重新输入！\n")
            
except serial.SerialException as e:
    print(f"❌ 串口打不开，请检查端口号是否正确或被占用: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("👋 串口已关闭。")