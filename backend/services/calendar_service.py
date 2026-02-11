from __future__ import annotations

class CalendarService:
    def __init__(self) -> None:
        self._appointments: list[dict[str, str]] = []

    def check_calendar(self, date: str) -> dict[str, object]:
        # Demo: fixed availability
        print(f"tool: calendar checked with date = {date}")
        return {
            "date": date,
            "available": ["3:00 PM", "4:00 PM"],
        }

    def book_appointment(self, date: str, time: str, name: str) -> dict[str, object]:
        print(f"tool: appointment booked with date = {date}, time = {time}, name = {name}")
        appt = {"date": date, "time": time, "name": name}
        self._appointments.append(appt)
        return {
            "status": "confirmed",
            "appointment": appt,
        }
