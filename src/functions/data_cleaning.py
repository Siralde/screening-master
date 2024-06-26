import pandas as pd
from datetime import datetime
import csv
import math
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#NOTE: The following CSV's are available but currently not used:
"""
- organization_descriptions.csv
- org_parents.csv
- category_groups.csv
- people_descriptions.csv
- jobs.csv
- investors.csv
- investment_partners.csv
- funds.csv
- events.csv
- event_appearances.csv
"""

def clean_organization_csv(organization_path,
                           start_date,
                           end_date):
    org_df = pd.read_csv(organization_path)

    if not start_date or end_date:
        start_date = datetime.strptime('2015-01-01', '%Y-%m-%d') # Start Day: 1 of January of 2015
        end_date = datetime.strptime('2018-12-31', '%Y-%m-%d') # End Day: 31 of December of 2018

    # Convert 'founded_on' to datetime
    org_df['founded_on'] = pd.to_datetime(org_df['founded_on'], errors='coerce')

    # Filter the dataframe by date range
    org_df = org_df[(org_df['founded_on'] >= start_date) & (org_df['founded_on'] <= end_date)]

    # Filter the dataframe where 'domain' is 'company'
    org_df = org_df[org_df['domain'] == 'company']

    # Specify the columns to drop
    columns_to_drop = ['type', 
        'cb_url', 
        'rank', 
        'created_at', 
        'updated_at', 
        'legal_name', 
        'roles', 
        'cb_url', 
        'rank', 
        'created_at', 
        'updated_at', 
        'legal_name', 
        'roles', 
        'email', 
        'phone', 
        'logo_url', 
        'alias1', 
        'alias2', 
        'alias3', 
        'primary_role', 
        'num_exits']

    # Drop the specified columns
    org_df = org_df.drop(columns=columns_to_drop)

    return org_df

def clean_funding_information_csv(org_df,
                                  funding_path,
                                  acquisitions_path,
                                  start_date,
                                  end_date):
    
    if not start_date or end_date:
        start_date = datetime.strptime('2015-01-01', '%Y-%m-%d') # Start Day: 1 of January of 2015
        end_date = datetime.strptime('2018-12-31', '%Y-%m-%d') # End Day: 31 of December of 2018

    # Grab the funding data: (IPO/Acquisition/Closure)
    fund_df = pd.read_csv(funding_path)

    # Merge the two dataframes
    merged_df = org_df.merge(fund_df, left_on='uuid', right_on='org_uuid', how="left")

    # Remove ones closed during warmup window
    merged_df['closed_on'] = pd.to_datetime(merged_df['closed_on'])
    filtered_close_merged_df = merged_df[(merged_df['closed_on'] < start_date) | (merged_df['closed_on'] > end_date) | (merged_df['closed_on'].isna())]

    # Remove ones acquired during warmup window
    ac_df = pd.read_csv(acquisitions_path)
    ac_df =  filtered_close_merged_df.merge(ac_df, left_on='uuid_x', right_on='acquiree_uuid', how="left")

    ac_df['acquired_on'] = pd.to_datetime(ac_df['acquired_on'])
    filter_ac_df = ac_df[(ac_df['acquired_on'] < start_date) | (ac_df['acquired_on'] > end_date) | (ac_df['acquired_on'].isna())]

    return filter_ac_df

def clean_ipos_csv(ipos_path,
                   filter_ac_df,
                   start_date,
                   end_date,
                   sim_start_date,
                   sim_end_date):
    
    if not start_date or end_date:
        start_date = datetime.strptime('2015-01-01', '%Y-%m-%d') # Start Day: 1 of January of 2015
        end_date = datetime.strptime('2018-12-31', '%Y-%m-%d') # End Day: 31 of December of 2018
    if not sim_start_date or sim_end_date:
        sim_start_date = datetime.strptime('2019-01-01', '%Y-%m-%d') # Start Day: 1 of January of 2019
        sim_end_date = datetime.strptime('2022-12-31', '%Y-%m-%d') # End Day: 31 of December of 2022


    # Remove ones that IPO'd during warmup window
    ipo_df = pd.read_csv(ipos_path)

    filter_ac_df.rename(columns={'uuid_x': 'uuid_org'}, inplace=True)
    filter_ac_df.rename(columns={'name_x': 'name_org'}, inplace=True)
    filter_ac_df.rename(columns={'permalink_x': 'permalink_org'}, inplace=True)

    ipo_df.rename(columns={'org_uuid': 'uuid_ipos'}, inplace=True)

    ipo_df = filter_ac_df.merge(ipo_df, left_on='uuid_org', right_on='uuid_ipos', how="left")

    ipo_df['went_public_on'] = pd.to_datetime(ipo_df['went_public_on'])
    AC_CL_IPO_df = ipo_df[(ipo_df['went_public_on'] < start_date) | (ipo_df['went_public_on'] > end_date) | (ipo_df['went_public_on'].isna())]

    # Filter companies > series B
    filter = ['series_b', 'series_c', 'series_d', 'series_e', 'series_f', 'series_g']
    organizations_to_remove = AC_CL_IPO_df[AC_CL_IPO_df['investment_type'].isin(filter)]['org_uuid'].unique()

    filtered_df = AC_CL_IPO_df[~AC_CL_IPO_df['org_uuid'].isin(organizations_to_remove)]
    
    # Remove duplicates
    unique_filtered = filtered_df.drop_duplicates(subset=['uuid_org'], keep='first')


    # Remove unwanted Columns
    columns_to_drop = ['uuid_y', 'name_y', 'type_x', 'permalink_y', 'cb_url_x', 'rank_x', 'created_at_x', 'updated_at_x', 'country_code_y', 'state_code_y', 'region_y', 'city_y', 'investment_type', 'announced_on', 'raised_amount_usd', 'raised_amount', 'raised_amount_currency_code', 'post_money_valuation_usd', 'post_money_valuation', 'post_money_valuation_currency_code', 'investor_count', 'org_uuid', 'org_name_x', 'lead_investor_uuids', 'uuid_x', 'name_x', 'type_y', 'permalink_x', 'cb_url_y', 'rank_y', 'created_at_y', 'updated_at_y', 'acquiree_uuid', 'acquiree_name', 'acquiree_cb_url', 'acquiree_country_code', 'acquiree_state_code', 'acquiree_region', 'acquiree_city', 'acquirer_uuid', 'acquirer_name', 'acquirer_cb_url', 'acquirer_country_code', 'acquirer_state_code', 'acquirer_region', 'acquirer_city', 'acquisition_type', 'acquired_on', 'price_usd', 'price', 'price_currency_code', 'uuid_y.1', 'name_y.1', 'type', 'permalink_y.1', 'cb_url', 'rank', 'created_at', 'updated_at', 'uuid_ipos', 'org_name_y', 'org_cb_url', 'country_code', 'state_code', 'region', 'city', 'stock_exchange_symbol', 'stock_symbol', 'went_public_on', 'share_price_usd', 'share_price', 'share_price_currency_code', 'valuation_price_usd', 'valuation_price', 'valuation_price_currency_code', 'money_raised_usd', 'money_raised', 'money_raised_currency_code']
    unique_filtered = unique_filtered.drop(columns=columns_to_drop, errors='ignore')

    unique_filtered.rename(columns={'country_code_x': 'country_code'}, inplace=True)
    unique_filtered.rename(columns={'state_code_x': 'state_code'}, inplace=True)
    unique_filtered.rename(columns={'region_x': 'region'}, inplace=True)
    unique_filtered.rename(columns={'city_x': 'city'}, inplace=True)

    # Filtering on date
    if not sim_start_date or sim_end_date:
        simulation_start_date = datetime.strptime('2019-01-01', '%Y-%m-%d') # Start Day: 1 of January of 2019
        simulation_end_date = datetime.strptime('2022-12-31', '%Y-%m-%d') # End Day: 31 of December of 2022

    unique_filtered['founded_on'] = pd.to_datetime(unique_filtered['founded_on'])

    unique_filtered['age_months'] = unique_filtered['founded_on'].apply(
        lambda x: math.ceil((simulation_start_date - x).days / 30) if pd.notnull(x) else float('nan')
    )
    # Convert URLs into binary variables
    unique_filtered['has_facebook_url'] = unique_filtered['facebook_url'].apply(has_url)
    unique_filtered['has_twitter_url'] = unique_filtered['twitter_url'].apply(has_url)
    unique_filtered['has_linkedin_url'] = unique_filtered['linkedin_url'].apply(has_url)

    return unique_filtered

def clean_funding_rounds_csv(funding_rounds_path,
                             sim_start_date,
                             unique_filtered
                            ):
    
    if not sim_start_date:
        sim_start_date = datetime.strptime('2019-01-01', '%Y-%m-%d') # Start Day: 1 of January of 2019


    funding_rounds_data = pd.read_csv(funding_rounds_path)

    # Convert relevant date columns to datetime format
    unique_filtered['founded_on'] = pd.to_datetime(unique_filtered['founded_on'])
    unique_filtered['last_funding_on'] = pd.to_datetime(unique_filtered['last_funding_on'])

    funding_rounds = funding_rounds_data[funding_rounds_data['org_uuid'].isin(unique_filtered['uuid_org'])]

    funding_rounds['announced_on'] = pd.to_datetime(funding_rounds['announced_on'])

    # Set the simulation start date (ts) to January 1, 2019
    ts = pd.to_datetime('2019-01-01')

    # 3. Filter the funding rounds that occurred before ts
    funding_before_ts = funding_rounds[funding_rounds['announced_on'] < ts]

    # 4. Count the number of funding rounds for each company
    round_count = funding_before_ts.groupby('org_uuid').size().reset_index(name='round_count')

    # 5. Sum up the raised amount in those rounds for each company
    raised_amount_usd = funding_before_ts.groupby('org_uuid')['raised_amount_usd'].sum().reset_index(name='raised_amount_usd')

    # Merge round count and raised amount into unique_filtered DataFrame
    unique_filtered = unique_filtered.merge(round_count, left_on="uuid_org", right_on='org_uuid', how='left', suffixes=('', '_x'))
    unique_filtered = unique_filtered.merge(raised_amount_usd, left_on="uuid_org", right_on='org_uuid', how='left', suffixes=('', '_y'))

    # Replace NaN values with 0 for companies with no data

    unique_filtered.fillna({'round_count':0}, inplace=True)

    # Ensure the column is treated as a numeric type
    # Convert all non-numeric values to NaN
    unique_filtered['raised_amount_usd'] = pd.to_numeric(unique_filtered['raised_amount_usd'], errors='coerce')

    # Fill NaN values with 0
    unique_filtered.fillna({'raised_amount_usd' : 0}, inplace=True)

    # Drop IDs
    unique_filtered = unique_filtered.drop(columns=['org_uuid', 'org_uuid_y'])

    # 2. Find the last funding round for each company within the Warmup window
    last_round_warmup = funding_before_ts.groupby('org_uuid').last()

    # 3. Extract the required data from the last funding round for each company
    last_round_investment_type = last_round_warmup['investment_type'].rename('last_round_investment_type')
    last_round_raised_amount_usd = last_round_warmup['raised_amount_usd'].rename('last_round_raised_amount_usd')
    last_round_post_money_valuation = last_round_warmup['post_money_valuation_usd'].rename('last_round_post_money_valuation')

    # 4. Calculate the time lapse in months between simulation start date and the last funding round
    last_round_timelapse_months = ((sim_start_date - last_round_warmup['announced_on']).dt.days / 30).apply(math.ceil).astype(int).rename('last_round_timelapse_months')

    unique_filtered = unique_filtered.merge(last_round_investment_type, left_on="uuid_org", right_on='org_uuid', how='left')
    unique_filtered = unique_filtered.merge(last_round_raised_amount_usd, left_on="uuid_org", right_on='org_uuid', how='left')
    unique_filtered = unique_filtered.merge(last_round_post_money_valuation, left_on="uuid_org", right_on='org_uuid', how='left')
    unique_filtered = unique_filtered.merge(last_round_timelapse_months, left_on="uuid_org", right_on='org_uuid', how='left')

    return (funding_before_ts, unique_filtered)

def clean_investments_csv(investments_path,
                        unique_filtered,
                        funding_before_ts):
    # Number of (unique) investors who participated in funding rounds during warmup

    invst_df = pd.read_csv(investments_path)

    # 1. Filter the investments that occurred during the Warmup window
    investments_warmup = invst_df[invst_df['funding_round_uuid'].isin(funding_before_ts['uuid'])]

    # 2. Calculate the number of unique investors for each company during the Warmup window
    investor_count = investments_warmup.groupby('funding_round_uuid')['investor_uuid'].nunique().rename('investor_count')

    # Merge investor count into funding_rounds DataFrame
    funding_before_ts = funding_before_ts.merge(investor_count, left_on="uuid", right_on="funding_round_uuid", how='left', suffixes=('', 'wup'))

    # 3. Filter the investments for the last funding round in the Warmup window for each company
    last_round_investments = investments_warmup.groupby('funding_round_uuid').last()

    # 4. Calculate the number of unique investors for each company in the last funding round
    last_round_investor_count = last_round_investments.groupby('funding_round_uuid')['investor_uuid'].nunique().rename('last_round_investor_count')

    # Merge last round investor count into funding_rounds DataFrame
    funding_before_ts = funding_before_ts.merge(last_round_investor_count, left_on="uuid", right_on="funding_round_uuid", how='left', suffixes=('', '_wup'))

    # Drop Duplicates
    funding_before_ts = funding_before_ts.sort_values(by=['org_uuid', 'announced_on'], ascending=[True, False])

    latest_funding_before_ts = funding_before_ts.drop_duplicates(subset='org_uuid', keep='first')

    # Create a DataFrame with only the required columns
    latest_funding_before_ts = latest_funding_before_ts[['org_uuid', 'investor_countwup', 'last_round_investor_count']]

    # Rename 'org_uuid' to 'uuid_org' to match the column name in unique_filtered
    latest_funding_before_ts.rename(columns={'org_uuid': 'uuid_org'}, inplace=True)

    # Merge the DataFrames
    unique_filtered = unique_filtered.merge(latest_funding_before_ts, on='uuid_org', how='left')

    # Replace NaN values with 0 for companies with no data
    unique_filtered.fillna(0, inplace=True)

    return unique_filtered

def clean_people_and_degrees_csv(people_path,
                     degrees_path,
                     unique_filtered):
    
    # Founders Data:

    ppl_df = pd.read_csv(people_path)
    people = ppl_df[ppl_df['featured_job_organization_uuid'].isin(unique_filtered['uuid_org'])]

    # Step 2: Define the regex pattern
    pattern = r'\b(cofounder|founder|ceo|cto|cmo|cpo|chief executive|chief technology|chief operation)\b'

    # Step 3: Filter the DataFrame to only include relevant rows
    filtered_people_data = people[people['featured_job_title'].apply(
        lambda x: bool(re.search(pattern, str(x), re.IGNORECASE))
    )]

    if filtered_people_data.empty:
        filtered_people_data = pd.DataFrame(columns=people.columns)

    # Step 4: Calculate founders_dif_country_count, founders_male_count, and founders_female_count
    founders_info = filtered_people_data.groupby('featured_job_organization_uuid').agg(
        founders_dif_country_count=('country_code', pd.Series.nunique),
        founders_male_count=('gender', lambda x: (x.str.lower() == 'male').sum()),
        founders_female_count=('gender', lambda x: (x.str.lower() == 'female').sum())
    ).reset_index()

    # Step 5: Merge the founders_info DataFrame back into the unique_filtered DataFrame
    unique_filtered = unique_filtered.merge(founders_info, left_on='uuid_org', right_on='featured_job_organization_uuid', how='left')

    # Fill NaN values with 0 for organizations with no founders information
    unique_filtered.fillna({'founders_dif_country_count': 0, 'founders_male_count': 0, 'founders_female_count': 0}, inplace=True)

    # Drop the redundant column after merging
    unique_filtered.drop(columns=['featured_job_organization_uuid'], inplace=True)
    
    
    # Look at education history

    degree_df = pd.read_csv(degrees_path)

    # Step 4: Merge the degrees data with the filtered people data
    merged_degrees = degree_df.merge(filtered_people_data, left_on='person_uuid', right_on='uuid', how='inner')
    # Step 5: Count the number of degrees for each founder
    degree_counts = merged_degrees.groupby('person_uuid').size().reset_index(name='degree_count')
    # Step 6: Merge the degree counts with the filtered people data to associate counts with organizations
    founders_with_degrees = filtered_people_data.merge(degree_counts, left_on='uuid', right_on='person_uuid', how='left')

    # Step 7: Replace NaN values in degree_count with 0
    founders_with_degrees.fillna({'degree_count':0}, inplace=True)

    # Step 8: Aggregate the degree counts at the organization level
    degree_stats = founders_with_degrees.groupby('featured_job_organization_uuid').agg(
        founders_degree_count_total=('degree_count', 'sum'),
        founders_degree_count_max=('degree_count', 'max'),
        founders_degree_count_mean=('degree_count', 'mean')
    ).reset_index()

    # Rename the column for clarity
    degree_stats.rename(columns={'featured_job_organization_uuid': 'uuid_org'}, inplace=True)

    # Step 9: Merge the degree statistics back into the unique_filtered DataFrame
    unique_filtered = unique_filtered.merge(degree_stats, on='uuid_org', how='left')

    # Fill NaN values with 0 for companies with no education data
    unique_filtered.fillna({'founders_degree_count_total': 0, 'founders_degree_count_max': 0, 'founders_degree_count_mean': 0}, inplace=True)

    return unique_filtered

def define_acquired(ac_df,
                    org_df,
                    sim_start_date,
                    sim_end_date):
    # Convert 'acquired_on' to datetime format
    ac_df['acquired_on'] = pd.to_datetime(ac_df['acquired_on'], errors='coerce')

    # Merge the datasets on 'uuid_org' and 'acquiree_uuid'
    merged_data = pd.merge(org_df, ac_df, left_on='uuid_org', right_on='acquiree_uuid', how='left', suffixes=('','_delete'))

    # Filter for companies acquired during the simulation window
    
    # Filtering on date
    if not sim_start_date or sim_end_date:
        sim_start_date = datetime.strptime('2019-01-01', '%Y-%m-%d') # Start Day: 1 of January of 2019
        sim_end_date = datetime.strptime('2022-12-31', '%Y-%m-%d') # End Day: 31 of December of 2022

    
    acquired_during_simulation = merged_data[
        (merged_data['acquired_on'] >= sim_start_date) & 
        (merged_data['acquired_on'] <= sim_end_date)
    ]

    # Create a new column 'outcome' and initialize it with 'NE' (No Event)
    org_df['outcome'] = 'NE'

    # Mark acquired companies as 'AC' in the 'outcome' column
    org_df.loc[org_df['uuid_org'].isin(acquired_during_simulation['uuid_org']), 'outcome'] = 'AC'

    return org_df, acquired_during_simulation

def define_ipo(ipo_df,
               org_df,
               sim_start_date,
               sim_end_date):
    
    if not sim_start_date or sim_end_date:
        sim_start_date = datetime.strptime('2019-01-01', '%Y-%m-%d') # Start Day: 1 of January of 2019
        sim_end_date = datetime.strptime('2022-12-31', '%Y-%m-%d') # End Day: 31 of December of 2022


    # Convert 'went_public_on' column to datetime
    ipo_df['went_public_on'] = pd.to_datetime(ipo_df['went_public_on'], errors='coerce')

    # Merge the datasets on 'uuid_org' and 'org_uuid'
    merged_ipos_data = pd.merge(org_df, ipo_df[['org_uuid', 'went_public_on']], left_on='uuid_org', right_on='org_uuid', how='left')

    # Filter for companies that went for an IPO during the simulation window
    ipo_during_simulation = merged_ipos_data[
        (merged_ipos_data['went_public_on'] >= sim_start_date) & 
        (merged_ipos_data['went_public_on'] <= sim_end_date)
    ]
    # Update 'outcome' column for IPO companies
    org_df.loc[org_df['uuid_org'].isin(ipo_during_simulation['uuid_org']), 'outcome'] = 'IP'

    return (org_df, ipo_during_simulation)

def define_fr(fund_df,
              org_df,
              sim_start_date,
              sim_end_date,
              ipo_during_simulation,
              acquired_during_simulation):
    
    if not sim_start_date or sim_end_date:
        sim_start_date = datetime.strptime('2019-01-01', '%Y-%m-%d') # Start Day: 1 of January of 2019
        sim_end_date = datetime.strptime('2022-12-31', '%Y-%m-%d') # End Day: 31 of December of 2022

    # Convert 'announced_on' column to datetime
    fund_df['announced_on'] = pd.to_datetime(fund_df['announced_on'], errors='coerce')

    # Merge the datasets on 'uuid_org' and 'org_uuid'
    merged_funding_data = pd.merge(org_df, fund_df[['org_uuid', 'announced_on']], left_on='uuid_org', right_on='org_uuid', how='left')

    # Filter for companies that had at least another round of funding during the simulation window
    funding_during_simulation = merged_funding_data[
        (merged_funding_data['announced_on'] >= sim_start_date) & 
        (merged_funding_data['announced_on'] <= sim_end_date)
    ]

    # Ensure that the outcome is not already "IP" or "AC"
    funding_during_simulation = funding_during_simulation[
        ~funding_during_simulation['uuid_org'].isin(ipo_during_simulation['uuid_org']) &
        ~funding_during_simulation['uuid_org'].isin(acquired_during_simulation['uuid_org'])
    ]

    # Update 'outcome' column for funding round companies
    org_df.loc[org_df['uuid_org'].isin(funding_during_simulation['uuid_org']), 'outcome'] = 'FR'

    return org_df

def define_cl(org_df,
              sim_start_date,
              sim_end_date):
    
    if not sim_start_date or sim_end_date:
        sim_start_date = datetime.strptime('2019-01-01', '%Y-%m-%d') # Start Day: 1 of January of 2019
        sim_end_date = datetime.strptime('2022-12-31', '%Y-%m-%d') # End Day: 31 of December of 2022

    # Convert 'closed_on' column to datetime
    org_df['closed_on'] = pd.to_datetime(org_df['closed_on'], errors='coerce')

    # Filter for companies that were closed during the simulation window
    closed_during_simulation = org_df[
        (org_df['closed_on'] >= sim_start_date) & 
        (org_df['closed_on'] <= sim_end_date)
    ]

    # Ensure that the outcome for companies already marked as "AC" remains "AC"
    closed_during_simulation = closed_during_simulation[
        org_df['outcome'] != 'AC'
    ]

    # Update 'outcome' column for closed companies
    org_df.loc[org_df['uuid_org'].isin(closed_during_simulation['uuid_org']), 'outcome'] = 'CL'

def clean_data(organization_path, 
               funding_rounds_path,
               acquisitions_path,
               ipos_path,
               investments_path,
               people_path,
               degrees_path,
               start_date, 
               end_date,
               sim_start_date,
               sim_end_date,
               ):
    
    # Filtering organizations.csv
    logger.info("Cleaning organizations.csv")
    org_df = clean_organization_csv(organization_path, start_date, end_date)
    

    # Filtering funding_rounds.csv and acquisitions.csv
    logger.info("Cleaning funding_rounds.csv and acquisitions.csv")
    filter_ac_df = clean_funding_information_csv(org_df,
                                                funding_rounds_path,
                                                acquisitions_path,
                                                start_date,
                                                end_date)

    # Filtering ipos.csv
    logger.info("Cleaning ipos.csv")
    unique_filtered = clean_ipos_csv(ipos_path,
                                     filter_ac_df,
                                     start_date,
                                     end_date,
                                     sim_start_date,
                                     sim_end_date)

    # Filtering  funding_rounds.csv
    logger.info("Cleaning funding_rounds.csv")
    funding_before_ts, unique_filtered = clean_funding_rounds_csv(funding_rounds_path,
                                                sim_start_date,
                                                unique_filtered
                                            )


    # Filtering  investments.csv
    logger.info("Cleaning investments.csv")
    unique_filtered = clean_investments_csv(investments_path,
                                          unique_filtered,
                                          funding_before_ts)

    # Filtering people.csv and degrees.csv
    logger.info("Cleaning people.csv and degrees.csv")
    unqiue_filtered = clean_people_and_degrees_csv( people_path,
                                                    degrees_path,
                                                    unique_filtered)


    # Begin to define Targets:
    
    logger.info("Defining targets")
    org_df = unique_filtered
    ac_df = pd.read_csv(acquisitions_path)
    ipo_df = pd.read_csv(ipos_path)
    fund_df = pd.read_csv(funding_rounds_path)


    ### DEFINING ACQUIRED (AC) ###
    logging.info("Defining ACs")
    org_df, acquired_during_simulation = define_acquired(ac_df,
                             org_df,
                             sim_start_date,
                             sim_end_date)

    ### DEFINING IPO (IP) ###
    logging.info("Defining IPs")

    org_df, ipo_during_simulation = define_ipo(ipo_df,
                        org_df,
                        sim_start_date,
                        sim_end_date)


    ## DEFINING FUNDING ROUND (FR) ###
    logging.info("Defining FRs")

    org_df = define_fr(fund_df,
                       org_df,
                       sim_start_date,
                       sim_end_date,
                       ipo_during_simulation,
                       acquired_during_simulation)

    ## DEFINING CLOSED (CL) ###
    logging.info("Defining CLs")

    org_df = define_cl(org_df,
                    sim_start_date,
                    sim_end_date)


    return org_df


def has_url(url):
    return 1 if pd.notnull(url) else 0