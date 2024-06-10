import asyncio
import requests
import pandas as pd
import joblib
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import string

async def send_photo_from_url(update: Update, photo_url: str) -> None:
    await update.message.reply_photo(photo=photo_url)

# Tạo danh sách key
free_key = "FREEKEY123"
permanent_keys = ["PERMKEY" + ''.join(random.choices(string.ascii_letters + string.digits, k=10)) for _ in range(10)]

# Lưu trữ thông tin sử dụng key
used_keys = {}
telegram_users = {}

admin_token = "ADM:HAGSUAJSH6SF6A777A"

def fetch_historical_results(limit=200):
    url = "https://m.coinvid.com/api/rocket-api/game/issue-result/page"
    params = {"subServiceCode": "RG1M", "size": limit}

 headers = {
        "Host": "m.coinvid.com",
        "Connection": "keep-alive",
        "sec-ch-ua": '"Chromium";v="124", "Android WebView";v="124", "Not-A.Brand";v="99"',
        "user_type": "rocket",
        "Blade-Auth": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJpc3N1c2VyIiwiYXVkIjoiYXVkaWVuY2UiLCJ0ZW5hbnRfaWQiOiI2NzczNDMiLCJ1c2VyX25hbWUiOiJoYWlhbmhuZTI2IiwidG9rZW5fdHlwZSI6ImFjY2Vzc190b2tlbiIsInJvbGVfbmFtZSI6IiIsInVzZXJfdHlwZSI6InJvY2tldCIsInVzZXJfaWQiOiIxNjI0NzQ4OTg5OTY5NDA4MDAxIiwiZGV0YWlsIjp7ImF2YXRhciI6IjIwIiwidmlwTGV2ZWwiOjJ9LCJhY2NvdW50IjoiaGFpYW5obmUyNiIsImNsaWVudF9pZCI6InJvY2tldF93ZWIiLCJleHAiOjE3MTg2MzU1OTEsIm5iZiI6MTcxODAzMDc5MX0.dckcPAbgwZRoDJH7_HRNPqt-1LlEYg1XZecRLb6cWYkByCAYAolrqA1LQKPXKn4YI20jmhTq6K0Fl75ugefwEA",
        "Accept-Language": "en-US",
        "sec-ch-ua-mobile": "?1",
        "Authorization": "Basic cm9ja2V0X3dlYjpyb2NrZXRfd2Vi",
        "User-Agent": "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.91 Mobile Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "sec-ch-ua-platform": "Android",
        "X-Requested-With": "mark.via.gp",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Referer": "https://m.coinvid.com/game/guessMain?gameName=RG1M&returnUrl=%2FgameList",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Cookie": "_fbp=fb.1.1717773039503.427590769630604432; JSESSIONID=VGBpK85oOB1D_a65GnuB4_V8kXvAVcbvED5Wlxkl"
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'data' in data and 'records' in data['data']:
            return data['data']['records']
    return []

def preprocess_data(records):
    df = pd.DataFrame(records)
    df['color'] = df['simpleResultFormatList'].apply(lambda x: x[0].get('color') if x else None)
    df['color'] = df['color'].map({'green': 0, 'red': 1})
    df['issue'] = pd.to_numeric(df['issue'], errors='coerce')
    df = df.dropna(subset=['issue', 'color'])

    # Thêm các tính năng mới
    df['previous_color'] = df['color'].shift(-1)
    df['color_change'] = (df['color'] != df['previous_color']).astype(int)
    df['issue_diff'] = df['issue'].diff().fillna(0)

    df = df.dropna(subset=['previous_color'])

    return df[['issue', 'color', 'previous_color', 'color_change', 'issue_diff']]

def train_model(df, model_type='xgboost'):
    X = df[['issue', 'previous_color', 'color_change', 'issue_diff']]
    y = df['color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'xgboost':
        model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    elif model_type == 'svm':
        model = SVC(probability=True, random_state=42)
    else:
        raise ValueError("Invalid model type. Choose from 'xgboost', 'random_forest', 'gradient_boosting', 'svm'.")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Model: {model_type}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return model

def predict_next_result(model, next_issue, last_color, last_color_change, last_issue_diff):
    prediction = model.predict([[next_issue, last_color, last_color_change, last_issue_diff]])
    return 'XANH' if prediction[0] == 0 else 'ĐỎ'

def load_model():
    try:
        model = joblib.load('model.pkl')
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print("No model found. Training a new one.")
        return None

def save_model(model):
    joblib.dump(model, 'model.pkl')
    print("Model saved successfully.")

def check_key(key, user_id):
    global used_keys, telegram_users

    # Check if the key is the admin key
    if key == admin_token:
        return True

    # Check if the key is a permanent key
    if key in permanent_keys:
        if key not in used_keys:
            used_keys[key] = user_id
            return True
        elif used_keys[key] == user_id:
            return True
        else:
            return False

    # Check if the key is the free key
    elif key == free_key:
        if user_id not in telegram_users:
            telegram_users[user_id] = datetime.now()
            return True
        else:
            time_used = datetime.now() - telegram_users[user_id]
            if time_used < timedelta(minutes=15):
                return True
            else:
                return False

    return False

async def start(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id
    key = context.args[0] if context.args else None

    if not key or not check_key(key, user_id):
        await update.message.reply_text("Key không hợp lệ hoặc đã hết hạn.")
        return

    context.user_data['last_issue'] = None
    context.user_data['last_color'] = None
    context.user_data['last_color_change'] = None
    context.user_data['last_issue_diff'] = None
    context.user_data['model'] = load_model()
    context.user_data['predictions_count'] = 0
    context.user_data['correct_predictions'] = 0
    context.user_data['running'] = True
    context.user_data['result_printed'] = False
    context.user_data['last_prediction'] = None

    chaohoi = """────────────────────────────────────────
=>TOOL DỰ ĐOÁN PHIÊN XANH ĐỎ COINVID VER 1.0
=>ADMIN: HOANG HAI X HAI ANH
────────────────────────────────────────"""

    await update.message.reply_text(chaohoi)
    await send_photo_from_url(update, "https://i.ibb.co/HPwS3pB/447685774-436119322549760-4433388996537746089-n.jpg")
    await update.message.reply_text("『Bot đã hoạt động, bắt đầu dự đoán phiên...』")
    await asyncio.sleep(2)

    while context.user_data['running']:
        try:
            if key == free_key and datetime.now() - telegram_users[user_id] > timedelta(minutes=15):
                await update.message.reply_text("Thời gian sử dụng key miễn phí đã hết.")
                context.user_data['running'] = False
                return

            historical_results = fetch_historical_results(limit=100)
            if historical_results:
                df = preprocess_data(historical_results)
                last_result = df.iloc[0]
                issue = last_result['issue']
                color = 'XANH' if last_result['color'] == 0 else 'ĐỎ'
                color_change = last_result['color_change']
                issue_diff = last_result['issue_diff']

                if context.user_data['last_issue'] is None or issue != context.user_data['last_issue']:
                    if context.user_data['last_issue'] is not None and issue != context.user_data['last_issue']:
                        if context.user_data['last_prediction'] is not None:
                            result_text = f"Phiên : {issue} | Kết quả : {color}"
                            if context.user_data['last_prediction'] == color:
                                result_text += " | WIN ✅"
                                context.user_data['correct_predictions'] += 1
                            else:
                                result_text += " | LOSE ❎"
                            await update.message.reply_text(result_text)

                        next_issue = issue + 1
                        next_prediction = predict_next_result(
                            context.user_data['model'], next_issue,
                            last_result['previous_color'],
                            color_change,
                            issue_diff
                        )
                        context.user_data['last_prediction'] = next_prediction
                        prediction_text = f"💡 Dự đoán phiên tiếp theo: {next_prediction}"
                        await update.message.reply_text(prediction_text)

                        context.user_data['predictions_count'] += 1
                        if context.user_data['predictions_count'] == 10:
                            accuracy = context.user_data['correct_predictions'] / context.user_data['predictions_count']
                            await update.message.reply_text(f"Tỷ lệ thắng sau 10 phiên là: {accuracy * 100:.2f}%")
                            context.user_data['predictions_count'] = 0
                            context.user_data['correct_predictions'] = 0

                    context.user_data['last_issue'] = issue
                    context.user_data['last_color'] = last_result['color']
                    context.user_data['last_color_change'] = color_change
                    context.user_data['last_issue_diff'] = issue_diff

                else:
                    context.user_data['result_printed'] = False

            else:
                await update.message.reply_text("Không có kết quả lịch sử để phân tích.")

        except Exception as e:
            await update.message.reply_text(f"Lỗi: {e}")

        await asyncio.sleep(5)

async def stop(update: Update, context: CallbackContext) -> None:
    context.user_data['running'] = False
    await update.message.reply_text("Bot stopped!")

async def show_keys(update: Update, context: CallbackContext) -> None:
    token = context.args[0] if context.args else None
    if token == admin_token:
        keys_text = "Danh sách key đã tạo:\n"
        keys_text += f"Free key: {free_key}\n"
        keys_text += "Permanent keys:\n" + "\n".join(permanent_keys)
        await update.message.reply_text(keys_text)
    else:
        await update.message.reply_text("Bạn không có quyền truy cập vào lệnh này.")

def main():
    application = Application.builder().token("7050851037:AAFOT2fxogbG383ubAIMcsA5Jfjuhk8jZVk").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("key", show_keys))

    application.run_polling()

if __name__ == "__main__":
    main()
