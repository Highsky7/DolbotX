#!/usr/bin/env python3
"""
(수정) bag recorder timestamp 사용 + dt=0 방어
"""

from pathlib import Path
import argparse, sys, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import rclpy.serialization as ser
from sensor_msgs.msg import Imu

DEFAULT_ROOT  = Path.home()/ "workspace"/"bag"/"gps_imu_car"
DEFAULT_BAG   = "konkuk_250721"
DEFAULT_TOPIC = "/imu/data"

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-b","--bag",  default=DEFAULT_BAG)
    ap.add_argument("-t","--topic",default=DEFAULT_TOPIC)
    ap.add_argument("--fmin", type=float, default=0.5)
    ap.add_argument("--fmax", type=float, default=10.0)
    ap.add_argument("--margin", type=float, default=1.10)
    return ap.parse_args()

def open_bag(path):
    reader=SequentialReader()
    reader.open(StorageOptions(uri=str(path),storage_id='sqlite3'),
                ConverterOptions('cdr','cdr'))
    return reader

def extract(reader, topic):
    ts, acc = [], []
    while reader.has_next():
        top, raw, t = reader.read_next()
        if top!=topic: continue
        msg=Imu(); ser.deserialize_message(raw,msg)
        ts.append(t*1e-9)                      # <-- bag recorder stamp
        acc.append([msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z])
    return np.asarray(ts), np.asarray(acc)

def cutoff(t, acc, fmin, fmax, margin):
    t-=t[0]
    dif=np.diff(t)
    dif=dif[dif>0]            # 0 제거
    if dif.size==0:
        sys.exit("[ERROR] all timestamp differences are 0; cannot compute Fs.")
    dt=np.median(dif)
    Fs=1/dt
    N=len(t); freq=fftfreq(N,dt); pos=freq>=0
    peaks=[]
    for i in range(3):
        S=np.abs(fft(acc[:,i]))
        band=(freq>=fmin)&(freq<=fmax)&pos
        pk = freq[band][np.argmax(S[band])] if np.any(band) else np.nan
        peaks.append(pk)
    fc=margin*np.nanmax(peaks)
    return fc,Fs,peaks

def main():
    args=parse()
    bag=Path(args.bag)
    if not bag.is_absolute(): bag=DEFAULT_ROOT/bag
    if not bag.is_dir(): sys.exit(f"bag not found: {bag}")
    print(f"[INFO] bag  : {bag}\n[INFO] topic: {args.topic}")

    reader=open_bag(bag)
    t,acc=extract(reader,args.topic)
    if t.size<10: sys.exit("too few IMU messages.")

    fc,Fs,pk=cutoff(t,acc,args.fmin,args.fmax,args.margin)
    print(f"[RESULT] peaks Hz  ax={pk[0]:.2f} ay={pk[1]:.2f} az={pk[2]:.2f}")
    print(f"[RESULT] 추천 lpf_cutoff = {fc:.2f} Hz")

    # optional plot
    try:
        import matplotlib.pyplot as plt
        t-=t[0]; dt=np.median(np.diff(t)); N=len(t)
        freq=fftfreq(N,dt); pos=freq>=0
        plt.figure(figsize=(8,4))
        plt.plot(freq[pos], np.abs(fft(acc[:,0]))[pos],'r')
        plt.xlim(0,15); plt.xlabel("Hz"); plt.ylabel("|FFT a_x|")
        plt.axvline(fc/args.margin,color='gray',ls='--')
        plt.axvline(fc,color='k',ls='--'); plt.title("Accel X spectrum")
        plt.show()
    except Exception as e:
        print("[WARN] plotting skipped:",e)

if __name__=="__main__":
    main()
