"""力传感器 MODBUS-RTU 独立连接测试。

打开串口，连续读取六通道力数据并打印。超时或 CRC 失败会抛异常。

用法:
    python scripts/test_force.py [--port COM5] [--reads 5] [--debug]

退出码: 0=通过, 1=失败.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_collection.config import ForceConfig  # noqa: E402
from data_collection.sensor_collectors.modbus_client import (  # noqa: E402
    ModbusClient,
    ModbusException,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="力传感器 MODBUS-RTU 连接测试")
    parser.add_argument("--port", type=str, default=None, help="串口，如 COM5")
    parser.add_argument("--reads", type=int, default=5, help="连续读取次数")
    parser.add_argument("--debug", action="store_true", help="打印原始通信帧")
    args = parser.parse_args()

    cfg = ForceConfig()
    if args.port:
        cfg.serial_port = args.port

    client = ModbusClient(
        port=cfg.serial_port,
        baudrate=cfg.baudrate,
        data_bits=cfg.data_bits,
        stop_bits=cfg.stop_bits,
        parity=cfg.parity,
        timeout=cfg.timeout,
        rs485_mode=cfg.rs485_mode,
        rs485_tx_pin=cfg.rs485_tx_pin,
        rs485_tx_level=cfg.rs485_tx_level,
        debug=args.debug,
    )

    try:
        for i in range(args.reads):
            raw = client.read_32bit_values(
                cfg.slave_address, cfg.reg_start_address, cfg.channel_count, signed=True
            )
            scaled = [v * cfg.scale_factor for v in raw]
            print(f"[INFO] 第{i + 1}次读取: {[f'{v:.2f}' for v in scaled]}")
            if i < args.reads - 1:
                time.sleep(cfg.sample_interval_ms / 1000.0)
    except ModbusException as e:
        print(f"[FAIL] MODBUS 读取失败: {e}")
        print("       排查: 串口号 / RS485 方向 / 从机地址 / 波特率；可用 --debug 看原始帧")
        return 1
    except Exception as e:
        print(f"[FAIL] 串口错误: {e} (检查 {cfg.serial_port} 是否被占用)")
        return 1
    finally:
        client.close()

    print(f"[PASS] 力传感器测试通过 ({cfg.serial_port}, slave=0x{cfg.slave_address:02X})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
