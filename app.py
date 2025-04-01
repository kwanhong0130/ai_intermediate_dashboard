import streamlit as st # web development
import numpy as np # np mean, np random 
import pandas as pd # read csv, df manipulation
import requests
import json
import time
import os
import concurrent.futures
import plotly.express as px

from loguru import logger
from datetime import datetime
from pytz import timezone
from streamlit_extras.metric_cards import style_metric_cards

lg_account_ids = {
    "lgairesearch": {"name": "LG AI연구원", "org_id": 886}, # LG AI연구원
    "lgcns-ai": {"name": "LG CNS", "org_id": 1036}, # LG CNS
    "dno-ai": {"name": "D&O", "org_id": 1040}, # D&O
    "lgeri-ai": {"name": "LG 경영연구원", "org_id": 1042}, # LG경영연구원
    "lgdisplay-ai": {"name": "LG 디스플레이", "org_id": 1045}, # LG디스플레이
    "lghnh-ai": {"name": "LG 생활건강", "org_id": 1035}, # LG 생활건강
    "lgensol-ai": {"name": "LG 에너지솔루션", "org_id": 1047}, # LG 에너지솔루션
    "lguplus-ai": {"name": "LG 유플러스", "org_id": 1044}, # LG유플러스
    "lginnotek-ai": {"name": "LG 이노텍", "org_id": 1038}, # LG이노텍
    "lgacademy-ai": {"name": "LG 인화원", "org_id": 1043}, # LG인화원
    "lge-ai": {"name": "LG 전자", "org_id": 1034}, # LG 전자
    "farmhannong-ai": {"name": "팜한농", "org_id": 1039}, # 팜한농
    "lghv-ai": {"name": "LG 헬로비전", "org_id": 1037}, # LG헬로비전
    "lgchem-airesearch": {"name": "LG 화학-AI", "org_id": 1046}, # LG 화학
    "lgchem": {"name": "LG 화학", "org_id": 281}, # LG 화학 2 별도 과정
    "hsad-ai": {"name": "LG HSAD", "org_id": 3551} # LG HSAD
}

base_url = "https://api-rest.elice.io"
auth_login_endpoint = "/global/auth/login/"
course_report_endpoint = "/global/organization/stats/course/report/request/"
remote_file_endpoint = "/global/remote_file/temp/get/"

now_datetime = datetime.now(timezone('Asia/Seoul'))
formatted_now_date = now_datetime.strftime("%Y%m%d_%H%M%S")

def get_auth_token(auth_url, post_data):
    response = requests.post(auth_url, data=post_data)
    res_json = response.json()
    logger.info(res_json)
    # Check the response status code
    if response.status_code == 200:
        # Response was successful, print the response content
        logger.info("Get auth token success")
    else:
        # Response was not successful, print the error message
        logger.error("Error: " + response.reason)
        logger.error("Auth Failed for some reason.")
    return res_json

auth_post_data = {
    "email": "kwanhong.lee@elicer.com",
    "password": "Younha486!!@@"
}

def request_track_report(endpoint, sessionkey, org_id, filter_cond={"$and":[]}):
    headers = {
        "Authorization": "Bearer " + sessionkey
    }

    # 2024년도 필터링 필요
    # 2023년도 {"$and":[{"begin_datetime":1672498800000},{"end_datetime":1704034800000}]}
    # 2024년도 {"$and":[{"begin_datetime":1704034800000},{"end_datetime":1735657200000}]}
    params = f"?organization_id={org_id}&filter_conditions={filter_cond}"
    request_url = base_url+endpoint+params

    logger.info(f"Request report url: {request_url}")

    response = requests.get(request_url, headers=headers)

    # Check the response status code
    if response.status_code == 200:
        # Response was successful, print the response content
        logger.info("Request of course report success")
        # st.write("트랙 리포트 다운로드 요청이 성공하였습니다. 🥳")
        res_json = response.json()
    else:
        # Response was not successful, print the error message
        logger.error("Error: " + response.reason)
        logger.error("Request failed for some reason.")

    return res_json['download_token']

def get_remote_file(endpoint, sessionkey, download_token):
    headers = {
        "Authorization": "Bearer " + sessionkey
    }

    params = f"?download_token={download_token}"
    request_url = base_url+endpoint+params
    response = requests.get(request_url, headers=headers)

    # Check the response status code
    if response.status_code == 200:
        # Response was successful, print the response content
        res_json = response.json()
        # logger.info(json.dumps(res_json))
    else:
        # Response was not successful, print the error message
        print("Error: " + response.reason)

    return res_json['url']

def cal_credit_usage_stats(report_filename):
    data_frame = pd.read_excel(report_filename, sheet_name=None, header=0)
    sheet_names = list(data_frame.keys())
    logger.info(sheet_names)

    # filter "student" records
    df = data_frame["종합"]
    logger.info(df.columns)
    # student_df = df[df['권한'] == 'student'] # 권한만료(nothing)도 필요?
    student_df = df[(df['권한'] == 'student') | (df['권한'] == 'nothing')]
    student_elicer_filtered_df = student_df[~student_df["이메일"].str.contains("elicer.com", na=False)]
    logger.info(student_elicer_filtered_df.head())

    # dataframe 학습진행율 10%이상 필터링
    # TODO: 이노텍 수강신청일 필터링 2월 1일~
    student_elicer_filtered_df = student_elicer_filtered_df.copy()
    student_elicer_filtered_df['학습진행률'] = student_elicer_filtered_df['학습진행률'].str.rstrip('%').astype('float') / 100.0
    filtered_df = student_elicer_filtered_df[student_elicer_filtered_df['학습진행률'] >= 0.1]
    count_over_10 = len(filtered_df)
    logger.info(f"Number of records with learning percent over 10%: {count_over_10}")

    os.remove(report_filename)

    return count_over_10

def fetch_data(url, acc_org_id, api_sessionkey):
    headers = {
        "Authorization": "Bearer " + api_sessionkey
    }

    download_token = None

    response = requests.get(url, headers=headers)
    res_json = response.json()
    # logger.info(res_json)
    # logger.info(response.reason)
    # Check the response status code
    if res_json['_result']['status'] == 'ok':
        # Response was successful, print the response content
        download_token = res_json['download_token']
    else:
        # Response was not successful, print the error message
        print("Error: " + response.reason)
    return download_token, acc_org_id

def _get_stats_result():
    # duration_filter_cond = json.dumps({"$and":[{"begin_datetime":1672498800000},
                                            #    {"end_datetime":1704034800000}]}) # 2023
    # duration_filter_cond = json.dumps({"$and":[{"begin_datetime":1704034800000},
                                            #    {"end_datetime":1735657200000}]}) # 2024.01.01 ~ 2024.12.31
    duration_filter_cond = json.dumps({"$and":[{"begin_datetime":1735657200000}]}) # 2025.01.01 ~                                        
    # report_download_token = request_track_report(course_report_endpoint, api_sessionkey, org_id,
                                                #  filter_cond=duration_filter_cond)
    # logger.info("Download token is: " + report_download_token)

    course_report_req_urls = []

    for key in lg_account_ids.keys():
        account_org_id = lg_account_ids[key]['org_id']
        # if lg-innotek, change duration_filter_cond
        # 2023.12.01 ~ 2025.01.01 {"$and":[{"begin_datetime":1701356400000},{"end_datetime":1735657200000}]}
        # TODO: 이노텍 23년 기정산 크레딧 제외 필요
        if account_org_id == 1038: # 이노텍
            duration_filter_cond = json.dumps({"$and":[{"begin_datetime":1734188400000}]})
        # elif account_org_id == 281: # 화학
            # duration_filter_cond = json.dumps({"$and":[{"begin_datetime":1704034800000},
                                                    #    {"end_datetime":1738335600000}]})
        # elif account_org_id == 1044: # 유플러스
            # duration_filter_cond = json.dumps({"$and":[{"begin_datetime":1704034800000},
                                                    #    {"end_datetime":1738335600000}]})
        elif account_org_id == 1034: # LG전자
            duration_filter_cond = json.dumps({"$and":[{"begin_datetime":1738335600000}]}) # 25.02.01~
        elif account_org_id == 281: # LG화학
            duration_filter_cond = json.dumps({"$and":[{"begin_datetime":1738335600000}]}) # 25.02.01~
        params = f"?organization_id={account_org_id}&filter_conditions={duration_filter_cond}"
        course_report_req_url = base_url+course_report_endpoint+params
        course_report_req_urls.append((course_report_req_url, account_org_id))

    get_auth_token_json = get_auth_token(base_url+auth_login_endpoint, auth_post_data)
    if get_auth_token_json['_result']['status'] == "ok":
        api_sessionkey = get_auth_token_json['sessionkey']
        logger.info("Sessionkey is: " + api_sessionkey)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit API calls concurrently
        futures = [executor.submit(fetch_data, url_info[0], url_info[1], api_sessionkey) for url_info in course_report_req_urls]

        # Wait for all results
        report_req_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Process the results
        for result in report_req_results:
            logger.info(f"Download token : org id {result[0]} : {result[1]}")


    progress_text = "요청한 리포트 파일의 생성과 다운로드를 진행중입니다. 🏄‍♂️"
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.5)
        my_bar.progress(percent_complete + 1, text=progress_text)

    # down_report_file_name = f"report_organization_{org_id}_{formatted_now_date}.xlsx"
    # It takes a while creating blob file
    # report_blob_url = get_remote_file(remote_file_endpoint, api_sessionkey, report_download_token)

    credit_usages = {}

    def _cal_credit_usage(report_req_result, api_sessionkey):
        if report_req_result[0] is not None:
            down_report_file_name = f"report_organization_{report_req_result[1]}_{formatted_now_date}.xlsx"
            report_blob_url = get_remote_file(remote_file_endpoint, api_sessionkey, report_req_result[0])

            if report_blob_url is not None:
                response = requests.get(report_blob_url)
                if response.status_code == 200:
                    with open(down_report_file_name, "wb") as f:
                        f.write(response.content)
                        logger.info("Report file is written as file")
                else:
                    logger.error("Error: " + response.reason)

            credit_usage = cal_credit_usage_stats(down_report_file_name)
            # credit_usages[str(report_req_result[1])] = credit_usage
        else:
            credit_usage = 0
            # credit_usages[str(report_req_result[1])] = 0
        
        return report_req_result[1], credit_usage

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit API calls concurrently
        futures = [executor.submit(_cal_credit_usage, result, api_sessionkey) for result in report_req_results]

        # Wait for all results
        credit_usage_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Process the results
        for credit_usage_result in credit_usage_results:
            credit_usages[str(credit_usage_result[0])] = credit_usage_result[1]

    # for result in report_req_results:
    #     if result[0] is not None:
    #         down_report_file_name = f"report_organization_{result[1]}_{formatted_now_date}.xlsx"
    #         report_blob_url = get_remote_file(remote_file_endpoint, api_sessionkey, result[0])

    #         if report_blob_url is not None:
    #             response = requests.get(report_blob_url)
    #             if response.status_code == 200:
    #                 with open(down_report_file_name, "wb") as f:
    #                     f.write(response.content)
    #                     logger.info("Report file is written as file")
    #             else:
    #                 logger.error("Error: " + response.reason)

    #         credit_usage = cal_credit_usage_stats(down_report_file_name)
    #         credit_usages[str(result[1])] = credit_usage
    #     else:
    #         credit_usages[str(result[1])] = 0

    my_bar.empty()

    return credit_usages


st.set_page_config(
    page_title = '2025 LG Intermediate Course 대시보드',
    page_icon = '📊',
    layout = 'wide'
)

st.markdown("""
<style>
div.stButton > button:first-child {
background-color: #A961DC; color:white;
}
</style>""", unsafe_allow_html=True)

# dashboard title
st.title("2025 LG Intermediate Course 크레딧 대시보드")

if st.button("대시보드 활성화 ✅"): 
    st.session_state.disabled = True
    credit_usages = _get_stats_result()

    account_names = []
    account_ids = []
    for lg_account in lg_account_ids.values():
        account_names.append(lg_account['name'])
        account_ids.append(lg_account['org_id'])

    arranged_credit_usages = []
    lg_innotek_used_credits = 0
    for org_id in account_ids:
        if org_id == 1038: # temp code to exclude lg innoteck used credit
            lg_innotek_used_credits = credit_usages[str(org_id)]
            # lg 이노텍 수강신청일 덮어쓰기 데이터 작업 필요(최초 수강신청일 기준 2월 이후)
            # lg_innotek_used_credits = credit_usages[str(org_id)] - 271 
            arranged_credit_usages.append(lg_innotek_used_credits)
            # arranged_credit_usages.append(0)
            # arranged_credit_usages.append(800)
        else:
            arranged_credit_usages.append(credit_usages[str(org_id)])

    result_df = pd.DataFrame({
        "Account": account_names,
        "Credit_Usage": arranged_credit_usages # [100] * len(lg_account_ids)
    })

    with st.container():
        col1, col2 = st.columns(2)

        # col1.metric(label="Gain", value=5000, delta=1000)
        # col2.metric(label="Loss", value=5000, delta=-1000)
        style_metric_cards()

        with col1:
            current_credit = 0
            # for credit in credit_usages.values():
                # current_credit += credit
            for credit in arranged_credit_usages:
                current_credit += credit
            # current_credit = current_credit - lg_innotek_used_credits # temp code to exclude lg innoteck used credit
            st.metric(label="사용 크레딧/총 크레딧", value=f"{current_credit}/5,000")

            st.dataframe(result_df, use_container_width=True)

        with col2:
            st.subheader("그룹사 별 크레딧 사용현황")
            # st.caption("LG 이노텍 사용 크레딧은 24년도 2월 1일 이후 수강신청 대상으로 집계되었습니다.")
            # st.caption("현재 LG 이노텍 사용 크레딧은 일시적으로 집계에 포함되어 있지 않으며, 4월 내 반영 예정입니다.")
            # st.caption("LG 이노텍의 AI연구원 사용 크레딧 800개 소진 반영")
            # st.caption("LG 화학 데이터 분석 중급 과정은 lgchem(LG 화학) 도메인에서 집계되었습니다.")
            # st.caption("LG 화학-엘리스 별도 콘텐츠 계약 건은 집계에서 제외되었습니다.")
            # num_account = len(lg_account_ids)
            # x_axis_names = [lg_account["name"] for lg_account in lg_account_ids.values()]
            # chart_data = pd.DataFrame(
            #     {
            #         "그룹사": x_axis_names,
            #         "크레딧 소진 수": [100] * len(lg_account_ids),
            #         "color": "#ffaa00"
            #     }
            # )
            # st.bar_chart(chart_data, x="그룹사", y="크레딧 소진 수", color="color")

            fig = px.bar(
                result_df,
                x="Account",
                y="Credit_Usage",
                text="Credit_Usage",
                labels={'Account': '그룹사', 'Credit_Usage':'크레딧 소진 양'},
                color = "Credit_Usage", 
                color_continuous_scale = 'viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
else: st.info("🔼 [대시보드 활성화 ✅] 버튼을 눌러주세요.")