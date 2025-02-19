# Описание гигафункций
 
giga_functions = [
    # описание гигафункции получения списка карт
    {
        "name": "get_cards",
        "description": "Возвращает список карт клиента",
        "parameters": {"type": "object", "properties": {}},
        "return_parameters": {
            "type": "object",
            "properties": {
                "cards": {
                    "type": "array",
                    "description": "Список карт клиента",
                    "properties": {},
                    "items": {
                        "type": "object",
                        "properties": {
                            "dpan": {
                                "type": "integer",
                                "description": "Четыре последние цифры номера карты",
                            },
                            "payment_system": {
                                "type": "string",
                                "description": "Платежная система карты",
                            },
                        },
                    },
                },
                "error": {
                    "type": "string",
                    "description": "Возвращается при возникновении ошибки, либо если карта не найдена покажет список доступных",
                },
            },
        },
    },
    # описание гигафункции получения баланса карты по ее последним четырем цифрам
    {
        "name": "get_balance",
        "description": "Возвращает баланс по карте по четырем последним цифрам номера карты",
        "parameters": {
            "type": "object",
            "properties": {
                "dpan": {
                    "type": "integer",
                    "description": "Четыре последние цифры номера карты",
                }
            },
            "required": ["dpan"],
        },
        "return_parameters": {
            "type": "object",
            "properties": {
                "dpan": {
                    "type": "integer",
                    "description": "Четыре последние цифры номера карты",
                },
                "balance": {"type": "integer", "description": "Баланс карты"},
                "error": {
                    "type": "string",
                    "description": "Возвращается при возникновении ошибки",
                },
            },
        },
    }
]
set_variable("giga_functions", giga_functions)
 
# Собираем описания функций для системного промпта
descriptions = [
    f"{func['name']}. Описание: {func['description']}" for func in giga_functions
]
giga_functions_description = "\n".join(descriptions)
 
set_variable("giga_functions_description", giga_functions_description)

