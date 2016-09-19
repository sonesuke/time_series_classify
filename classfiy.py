import numpy as np
import numpy.fft as fft
from numpy.random import *
import matplotlib.pyplot as plt
import pandas as pd


N = 1024
dt = 1.0 / N
f1 = 4.0
f2 = 0.5

def generate_ng():
    t = np.linspace(1, N, N)*dt - dt
    y = np.sin(2*np.pi*f1*t + rand() * 2 * np.pi) + rand(N) * 4
    hanning_window = np.hanning(N)
    f_y = fft.fft(y * hanning_window)
    f_y = fft.fft(y)
    frq = fft.fftfreq(N, dt)
    colname =  ["t" + str(i) for i in range(N)] + ["mark"]
    df = pd.DataFrame([np.append(abs(f_y), "NG")], columns=colname)
    return (t, y, frq, f_y, df)

def generate_ok():
    t = np.linspace(1, N, N)*dt - dt
    y = np.sin(2*np.pi*f2*t + rand() * 2 * np.pi) + rand(N) * 4
    hanning_window = np.hanning(N)
    f_y = fft.fft(y * hanning_window)
    f_y = fft.fft(y)
    frq = fft.fftfreq(N, dt)
    colname =  ["t" + str(i) for i in range(N)] + ["mark"]
    df = pd.DataFrame([np.append(abs(f_y), "OK")], columns=colname)
    return (t, y, frq, f_y, df)


print "generate train data"
df = pd.DataFrame()
for i in range(10):
    (t, y, frq, f_y, f) = generate_ng()
    df = df.append(f)

for i in range(10):
    (t, y, frq, f_y, f) = generate_ok()
    df = df.append(f)


from sklearn.metrics import confusion_matrix

print "generate test data"
test_df = pd.DataFrame()
for i in range(10):
    (t, y, frq, f_y, f) = generate_ng()
    test_df = test_df.append(f)

for i in range(10):
    (t, y, frq, f_y, f) = generate_ok()
    test_df = test_df.append(f)


print "random forest"
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(df.drop("mark", axis=1), df["mark"])
output = model.predict(test_df.drop("mark", axis=1))
print confusion_matrix(test_df["mark"], output)

print "SVM"
from sklearn.svm import LinearSVC
model = LinearSVC(C=1.0)
model.fit(df.drop("mark", axis=1), df["mark"])
output = model.predict(test_df.drop("mark", axis=1))
print confusion_matrix(test_df["mark"], output)


# plt.subplot(2, 1, 1)
# plt.plot(t, y)
# plt.subplot(2, 1, 2)
# plt.plot(frq, abs(f_y))
# plt.axis([0, N/2, 0, max(abs(f_y))])
# plt.show()
