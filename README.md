import tensorflow as tf
import numpy as np

# تحديد النموذج
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1)
])

# تحديد دالة الخسارة وطريقة التحسين
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

# بيانات الدخل
x_train = np.arange(1.00, 35.01, 0.01).reshape(-1, 1)

# بيانات الخرج
y_train = np.random.uniform(size=(len(x_train), 1))

# تدريب النموذج
model.fit(x_train, y_train, epochs=100)

# تنبؤ موعد الانفجار
prediction = model.predict(np.array([[35.0]]))
print(f"تنبؤ موعد الانفجار: {prediction[0][0]}")

# تحديد متغيرات لحفظ البيانات
last_ten_numbers = []
is_first_input = True

# الحصول على المدخلات وتدريب النموذج
while True:
    # الحصول على 10 أرقام من المستخدم
    inputs = []
    for i in range(10):
        num = int(input("ادخل رقمًا: "))
        inputs.append(num)
        last_ten_numbers.append(num)
        
    # تحديد بيانات الدخل والخرج
    x_train = np.array(last_ten_numbers).reshape(-1, 1)
    y_train = np.random.uniform(size=(len(x_train), 1))
    
    # تدريب النموذج
    model.fit(x_train, y_train, epochs=10)
    
    # تنبؤ موعد الانفجار القادم
    prediction = model.predict(np.array([[35.0]]))
    print(f"تنبؤ موعد الانفجار القادم: {prediction[0][0]}")
    
    # اعادة ضبط المتغيرات للحفاظ على اخر 10 ارقام فقط
    last_ten_numbers = last_ten_numbers[-10:]
    is_first_input = False
