#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-25 13:15:51
# @Author  : Lee (lijingyang@tcl.com)

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


# the source dir of datas
Original_corpus_dir = '/home/lee/workspace/cls-nlp_git/cls-nlp/cls_scene/corpus'

# the corpus dir for model
Corpus_dir = './corpus/'

"""
Define labels and datas for Demo v1
   for item in tup:
     item[0] included filesname
     item[1] represent numic label
     item[2] represent label name
"""

Data_tup_v1 = [(['play_control.data', 'tv_ui.data'], 0, 'UI'),

               (['tv_setting_and_state_query.data'], 1, 'Device'),

               (['tv_channel_switch.data'], 2, 'Channel'),

               (['cartoon.data', 'film.data', 'history.data',
                 'normal_video.data', 'tv_play.data', 'programme.data',
                 'acrobatics.data', 'comic_dialogue.data',
                 'short_sketch.data', 'traditional_opera.data'], 3, 'Media'),

               (['music_on_demand.data'], 4, 'Music'),

               (['singing_on_demand.data'], 5, 'Sing'),

               (['basic_rate.data', 'basic_weather.data',
                 'disaster_warn.data'], 6, 'Weather'),

               (['app_basic_control.data',
                 'app_download_install_uninstall.data',
                 'app_search.data'], 7, 'App'),

               (['video_interactive.data'], 8, 'Interactive'),

               (['stock_fund_information.data'], 9, 'Stock&Fund'),

               (['industry_news.data', 'joy_news.data',
                 'normal_news.data'], 10, 'News'),

               (['humanity_encyclopedia.data', 'science_encyclopedia.data'],
                11, 'Humanity&Science_Enc'),

               (['media_encyclopedia.data'], 12, 'Media_Enc'),

               (['music_interactive.data', 'music_encyclopedia.data'], 13, 'Music_Enc'),

               (['alarm_clock_setting.data'], 14, 'AlarmClock'),

               (['remind_setting.data'], 15, 'Remind'),

               (['calculate.data'], 16, 'Calculate'),

               (['unit_conversion.data'], 17, 'Unit_Conv'),

               (['currency_converter.data'], 18, 'Currency_Conv'),

               (['translation.data'], 19, 'Translate'),

               (['time_query.data', 'time_zone_query.data'], 20, 'Time'),

               (['map.data', 'navigation.data', 'traffic.data'], 21, 'Map'),

               (['cinema.data', 'other.data', 'playground.data',
                 'hosptial.data', 'hotel.data', 'mall.data',
                 'park.data', 'restaurant.data', 'school.data'], 22, 'Around'),

               (['joke.data'], 23, 'Joke'),

               (['poetry.data'], 24, 'Poetry')]


Data_tup_v2 = [(['play_control.data', 'tv_ui.data'], 0, 'UI'),

               (['tv_setting_and_state_query.data'], 1, 'Device'),

               (['tv_channel_switch.data'], 2, 'Channel'),

               (['cartoon.data', 'film.data', 'history.data',
                 'normal_video.data', 'tv_play.data', 'programme.data',
                 'acrobatics.data', 'comic_dialogue.data',
                 'short_sketch.data', 'traditional_opera.data'], 3, 'Media'),

               (['music_on_demand.data'], 4, 'Music'),

               (['singing_on_demand.data'], 5, 'Sing'),

               (['basic_rate.data', 'basic_weather.data',
                 'disaster_warn.data'], 6, 'Weather'),

               (['app_basic_control.data',
                 'app_download_install_uninstall.data',
                 'app_search.data'], 7, 'App'),

               (['stock_fund_information.data'], 8, 'Stock&Fund'),

               (['industry_news.data', 'joy_news.data',
                 'normal_news.data'], 9, 'News'),

               (['humanity_encyclopedia.data', 'science_encyclopedia.data'],
                10, 'Humanity&Science_Enc'),

               (['media_encyclopedia.data'], 11, 'Media_Enc'),

               (['music_interactive.data', 'music_encyclopedia.data'], 12, 'Music_Enc'),

               (['remind_setting.data'], 13, 'Remind'),

               (['calculate.data', 'unit_conversion.data',
                 'currency_converter.data'], 14, 'Calculate'),

               (['time_query.data', 'time_zone_query.data'], 15, 'Time'),

               (['map.data', 'navigation.data', 'traffic.data'], 16, 'Map'),

               (['cinema.data', 'other.data', 'playground.data',
                 'hosptial.data', 'hotel.data', 'mall.data',
                 'park.data', 'restaurant.data', 'school.data'], 17, 'Around'),

               (['poetry.data'], 18, 'Poetry')]
