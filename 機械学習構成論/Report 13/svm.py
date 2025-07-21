from libsvm.svmutil import *

def analyze_alpha(datafile, kernel_type=0):
    print(f"Processing {datafile} with kernel_type={kernel_type}")
    
    y, x = svm_read_problem(datafile)
    param = svm_parameter(f'-s 0 -t {kernel_type} -c 1 -q')
    prob = svm_problem(y, x)
    model = svm_train(prob, param)

    sv_coef = model.get_sv_coef()  # α * y
    SVs = model.get_SV()           # 支持向量本身

    print("Support Vector Coefficients (α * y):")
    for i, coef in enumerate(sv_coef):
        print(f"α[{i}] = {coef[0]}")
    print("\n")

# 分析三组数据，分别使用线性核（t=0）和RBF核（t=2）
for file in ['data1.txt', 'data2.txt', 'data3.txt']:
    analyze_alpha(file, kernel_type=0)  # linear kernel
    analyze_alpha(file, kernel_type=2)  # RBF kernel