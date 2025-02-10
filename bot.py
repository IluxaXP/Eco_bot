import os
import pandas as pd
import matplotlib.pyplot as plt
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from matplotlib.dates import DateFormatter
import asyncio
from matplotlib import rcParams
from prophet import Prophet
from data.TOKEN import API_TOKEN
import aiofiles

# Настройка шрифта для графиков
rcParams["font.family"] = "DejaVu Sans"
plt.style.use("seaborn-v0_8")

# Создание бота и диспетчера событий
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Кэш моделей для прогнозов
model_cache = {}
cache_ttl = 3600  # Время жизни кэша в секундах

# Чтение данных из CSV-файла
csv_file = "new_data.csv"
df = pd.read_csv(csv_file)
df["Period"] = pd.to_datetime(df["Period"], errors="coerce")  # Преобразование даты
# Удаление строк с отсутствующими критическими значениями
df = df.dropna(subset=["District", "Period", "MonthlyAverage", "PDKnorm", "Parameter"])

# Предобработка данных для прогнозов
preprocessed_data = {
    district: {
        param: group[["Period", "MonthlyAverage"]]
        .rename(columns={"Period": "ds", "MonthlyAverage": "y"})
        .sort_values("ds")
        for param, group in df[df["District"] == district].groupby("Parameter")
    }
    for district in df["District"].unique()
}

# Получение уникальных районов для клавиатуры
districts = df["District"].unique().tolist()
keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text=d)] for d in districts], resize_keyboard=True
)


# Функция для удаления выбросов из данных
def remove_outliers(df, sigma=3):
    """Удаление выбросов с использованием стандартного отклонения"""
    mean = df["y"].mean()
    std_dev = df["y"].std()
    return df[(df["y"] >= mean - sigma * std_dev) & (df["y"] <= mean + sigma * std_dev)]


# Асинхронная функция для создания прогноза и управления кэшем
async def create_forecast(district, param):
    """Асинхронное создание прогноза с кэшированием и управлением временем жизни кэша"""
    cache_key = f"{district}_{param}"

    # Проверка наличия прогноза в кэше и его актуальности
    if cache_key in model_cache:
        model, forecast, timestamp = model_cache[cache_key]
        if asyncio.get_event_loop().time() - timestamp < cache_ttl:
            return model, forecast

    data = preprocessed_data.get(district, {}).get(param)
    if data is None or len(data) < 24:
        return None, None

    try:
        clean_data = remove_outliers(data)

        # Создание и обучение модели Prophet
        model = Prophet(
            changepoint_prior_scale=0.15,
            seasonality_prior_scale=15,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.80,
        )
        model.fit(clean_data)

        # Построение будущих значений для прогноза
        future = model.make_future_dataframe(periods=24, freq="ME")
        forecast = model.predict(future)

        # Сохранение прогноза в кэш
        model_cache[cache_key] = (model, forecast, asyncio.get_event_loop().time())
        return model, forecast

    except Exception:
        return None, None


# Обработчик команды /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Выберите район:", reply_markup=keyboard)


# Обработчик выбора района и построения графиков
@dp.message(F.text.in_(districts))
async def send_combined_plot(message: types.Message):
    district = message.text

    if district not in preprocessed_data:
        return await message.answer("Данные по району не найдены")

    parameters = list(preprocessed_data[district].keys())
    if not parameters:
        return await message.answer("Нет данных по параметрам загрязнения")

    num_plots = len(parameters)
    fig, axs = plt.subplots(
        num_plots, 1, figsize=(12, 5 * num_plots) if num_plots > 1 else (12, 5)
    )
    plt.subplots_adjust(hspace=0.8)

    if num_plots == 1:
        axs = [axs]

    colors = plt.cm.tab10.colors
    caption = f"Анализ и прогноз для {district}"

    tasks = []
    for param in parameters:
        tasks.append(create_forecast(district, param))

    results = await asyncio.gather(*tasks)

    for i, ((model, forecast), param) in enumerate(zip(results, parameters)):
        ax = axs[i]
        color = colors[i % len(colors)]
        data = preprocessed_data[district][param]

        # Построение исторических данных
        ax.plot(
            data["ds"],
            data["y"],
            color=color,
            linewidth=1.5,
            label="Исторические данные",
        )

        if forecast is not None:
            forecast_period = forecast[forecast["ds"] > data["ds"].max()]
            ax.plot(
                forecast_period["ds"],
                forecast_period["yhat"],
                "purple",
                linestyle="--",
                label="Прогноз",
            )

        # Построение уровня ПДК
        pdk_value = df[(df["District"] == district) & (df["Parameter"] == param)][
            "PDKnorm"
        ].iloc[0]
        ax.axhline(pdk_value, color="red", linestyle="-.", label="ПДК")
        ax.set_title(f"{district} - {param}", fontsize=12, pad=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

    # Сохранение графика и отправка пользователю
    filename = f"{district}_forecast.png"
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close()

    # Асинхронная работа с файлами
    async with aiofiles.open(filename, mode="rb") as file:
        await message.answer_photo(
            types.BufferedInputFile(await file.read(), filename=filename),
            caption=caption,
        )

    os.remove(filename)  # Удаление временного файла


# Главная функция для запуска бота
async def main():
    await dp.start_polling(bot)


# Точка входа в приложение
if __name__ == "__main__":
    asyncio.run(main())
