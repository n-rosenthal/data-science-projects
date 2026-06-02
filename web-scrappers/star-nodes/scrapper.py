"""
   =star-nodes/scrapper.py= is a simple web-scrapper for planetary and astrological data

   @date 2026-05-26
   @version 1.0
   @author n-rosenthal
"""
import playwright

URL: str = r"https://horoscopes.astro-seek.com/calculate-planetary-hours/?narozeni_city=Porto+Alegre%2C+Brazil&narozeni_input_hidden=&narozeni_hidden_local_tz=1&narozeni_stat_hidden=BR&narozeni_podstat_hidden=Rio+Grande+do+Sul&narozeni_podstat_kratky_hidden=&narozeni_podstat2_kratky_hidden=&narozeni_tzid_id=181&narozeni_mesto_hidden=Porto+Alegre&narozeni_den=26&narozeni_mesic=05&narozeni_rok=2026&tolerance=1&narozeni_sirka_stupne=30&narozeni_sirka_minuty=2&narozeni_sirka_smer=1&narozeni_delka_stupne=51&narozeni_delka_minuty=14&narozeni_delka_smer=1#select_local_tz_anchor";

