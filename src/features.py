from __future__ import annotations
import pandas as pd, numpy as np
from pathlib import Path
def read(raw_dir, name):
    return pd.read_csv(Path(raw_dir) / name, low_memory=False)

def read_optional(raw_dir, name):
    """Load a CSV if it exists; return an empty DataFrame otherwise."""
    path = Path(raw_dir) / name
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()
def safe_div(a,b): return pd.Series(a).astype(float).divide(pd.Series(b).replace(0,np.nan).astype(float)).to_numpy()
def add_num(df, cols):
    for c in cols:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0)
    return df
def rolling_peak(df, group_col, year_col, cols, window, prefix, agg='sum'):
    d=df.sort_values([group_col,year_col]).copy()
    rows=[]
    for c in cols:
        if c not in d: continue
        if agg=='sum':
            r=d.groupby(group_col)[c].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        elif agg=='mean':
            r=d.groupby(group_col)[c].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        else:
            r=d.groupby(group_col)[c].rolling(window, min_periods=1).min().reset_index(level=0, drop=True)
        d[f'{prefix}_peak{window}_{c}']=r
        rows.append(f'{prefix}_peak{window}_{c}')
    return d.groupby(group_col)[rows].max().reset_index() if agg!='min' else d.groupby(group_col)[rows].min().reset_index()



def build_player_features(raw_dir: str | Path, processed_dir: str | Path | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir) if processed_dir is not None else None
    master=read(raw_dir, 'Master.csv'); batting=read(raw_dir, 'Batting.csv'); pitching=read(raw_dir, 'Pitching.csv'); fielding=read(raw_dir, 'Fielding.csv')
    hof=read(raw_dir, 'HallOfFame.csv'); awards=read(raw_dir, 'AwardsPlayers.csv'); awardshare=read(raw_dir, 'AwardsSharePlayers.csv'); allstar=read(raw_dir, 'AllstarFull.csv')
    salaries=read(raw_dir, 'Salaries.csv'); teams=read(raw_dir, 'Teams.csv'); batting_post=read(raw_dir, 'BattingPost.csv'); pitching_post=read(raw_dir, 'PitchingPost.csv'); fielding_of=read(raw_dir, 'FieldingOF.csv'); appearances=read_optional(raw_dir, 'Appearances.csv')
    # Master
    m=master.copy(); m['debut_dt']=pd.to_datetime(m['debut'],errors='coerce'); m['final_dt']=pd.to_datetime(m['finalGame'],errors='coerce')
    m['debut_year']=m['debut_dt'].dt.year; m['final_year']=m['final_dt'].dt.year
    m['debut_age']=m['debut_year']-m['birthYear']; m['final_age']=m['final_year']-m['birthYear']; m['career_span_years']=m['final_year']-m['debut_year']+1
    m['bmi']=703*m['weight']/np.square(m['height']); m['international_flag']=(m['birthCountry'].fillna('Unknown')!='USA').astype(int)
    m['bats_throws']=m['bats'].fillna('U')+'_'+m['throws'].fillna('U'); m['birth_month_sin']=np.sin(2*np.pi*m['birthMonth'].fillna(0)/12); m['birth_month_cos']=np.cos(2*np.pi*m['birthMonth'].fillna(0)/12)
    base=m[['playerID','nameFirst','nameLast','birthYear','birthMonth','birthCountry','weight','height','bats','throws','bats_throws','debut_year','final_year','debut_age','final_age','career_span_years','bmi','international_flag','birth_month_sin','birth_month_cos']].copy()
    base['player_name']=(base['nameFirst'].fillna('')+' '+base['nameLast'].fillna('')).str.strip()
    seasons=pd.concat([batting[['playerID','yearID']],pitching[['playerID','yearID']],fielding[['playerID','yearID']]],ignore_index=True).drop_duplicates()
    seas=seasons.groupby('playerID')['yearID'].agg(n_mlb_seasons='nunique',first_year='min',last_year='max').reset_index()
    base=base.merge(seas,on='playerID',how='left'); base['final_year']=base['final_year'].fillna(base['last_year']); base['debut_year']=base['debut_year'].fillna(base['first_year']); base['n_mlb_seasons']=base['n_mlb_seasons'].fillna(0)
    # BBWAA eligibility: retired at least 5 calendar years (first eligible ballot = final_year + 5).
    base['eligibility_year']=base['final_year']+5
    # Batting
    bat=add_num(batting.copy(), ['G','AB','R','H','2B','3B','HR','RBI','SB','CS','BB','SO','IBB','HBP','SH','SF','GIDP'])
    bat_year=bat.groupby(['playerID','yearID'],as_index=False)[['G','AB','R','H','2B','3B','HR','RBI','SB','CS','BB','SO','IBB','HBP','SH','SF','GIDP']].sum()
    bat_year['PA']=bat_year[['AB','BB','HBP','SH','SF']].sum(axis=1); bat_year['TB']=bat_year['H']+bat_year['2B']+2*bat_year['3B']+3*bat_year['HR']
    bat_year['BA']=safe_div(bat_year['H'],bat_year['AB']); bat_year['OBP']=safe_div(bat_year['H']+bat_year['BB']+bat_year['HBP'], bat_year['AB']+bat_year['BB']+bat_year['HBP']+bat_year['SF']); bat_year['SLG']=safe_div(bat_year['TB'],bat_year['AB']); bat_year['OPS']=bat_year['OBP']+bat_year['SLG']
    league=bat_year.groupby('yearID').agg(lg_AB=('AB','sum'),lg_H=('H','sum'),lg_2B=('2B','sum'),lg_3B=('3B','sum'),lg_HR=('HR','sum'),lg_BB=('BB','sum'),lg_HBP=('HBP','sum'),lg_SF=('SF','sum')).reset_index(); league['lg_TB']=league['lg_H']+league['lg_2B']+2*league['lg_3B']+3*league['lg_HR']; league['lg_OBP']=safe_div(league['lg_H']+league['lg_BB']+league['lg_HBP'],league['lg_AB']+league['lg_BB']+league['lg_HBP']+league['lg_SF']); league['lg_SLG']=safe_div(league['lg_TB'],league['lg_AB']); league['lg_OPS']=league['lg_OBP']+league['lg_SLG']
    stdops=bat_year[bat_year.PA>=100].groupby('yearID')['OPS'].std().rename('lg_OPS_sd').reset_index(); bat_year=bat_year.merge(league[['yearID','lg_OPS']],on='yearID',how='left').merge(stdops,on='yearID',how='left'); bat_year['OPS_plus_proxy']=100*pd.Series(bat_year['OPS']).divide(bat_year['lg_OPS'].replace(0,np.nan)); bat_year['OPS_z']=pd.Series(bat_year['OPS']-bat_year['lg_OPS']).divide(bat_year['lg_OPS_sd'].replace(0,np.nan))
    bat_c=bat_year.groupby('playerID')[['G','AB','R','H','2B','3B','HR','RBI','SB','CS','BB','SO','IBB','HBP','SH','SF','GIDP','PA','TB']].sum().add_prefix('bat_').reset_index()
    bat_c['bat_BA']=safe_div(bat_c['bat_H'],bat_c['bat_AB']); bat_c['bat_OBP']=safe_div(bat_c['bat_H']+bat_c['bat_BB']+bat_c['bat_HBP'],bat_c['bat_AB']+bat_c['bat_BB']+bat_c['bat_HBP']+bat_c['bat_SF']); bat_c['bat_SLG']=safe_div(bat_c['bat_TB'],bat_c['bat_AB']); bat_c['bat_OPS']=bat_c['bat_OBP']+bat_c['bat_SLG']; bat_c['bat_ISO']=bat_c['bat_SLG']-bat_c['bat_BA']; bat_c['bat_HR_rate']=safe_div(bat_c['bat_HR'],bat_c['bat_PA']); bat_c['bat_BB_rate']=safe_div(bat_c['bat_BB'],bat_c['bat_PA']); bat_c['bat_SO_rate']=safe_div(bat_c['bat_SO'],bat_c['bat_PA']); bat_c['bat_SB_success']=safe_div(bat_c['bat_SB'],bat_c['bat_SB']+bat_c['bat_CS'])
    era_bat=bat_year.groupby('playerID').agg(bat_mean_OPS_z=('OPS_z','mean'),bat_max_OPS_z=('OPS_z','max'),bat_mean_OPS_plus_proxy=('OPS_plus_proxy','mean'),bat_max_OPS_plus_proxy=('OPS_plus_proxy','max')).reset_index(); bat_c=bat_c.merge(era_bat,on='playerID',how='left')
    peak3=rolling_peak(bat_year,'playerID','yearID',['HR','H','RBI','SB'],3,'bat','sum'); peak3m=rolling_peak(bat_year,'playerID','yearID',['OPS','OPS_z','OPS_plus_proxy'],3,'bat','mean'); peak5=rolling_peak(bat_year,'playerID','yearID',['HR','H','RBI'],5,'bat','sum'); peak5m=rolling_peak(bat_year,'playerID','yearID',['OPS','OPS_z','OPS_plus_proxy'],5,'bat','mean')
    for df in [peak3,peak3m,peak5,peak5m]: bat_c=bat_c.merge(df,on='playerID',how='left')
    # Pitching
    pit=add_num(pitching.copy(), ['W','L','G','GS','CG','SHO','SV','IPouts','H','ER','HR','BB','SO','IBB','WP','HBP','BK','BFP','GF','R','SH','SF','GIDP'])
    pit_year=pit.groupby(['playerID','yearID'],as_index=False)[['W','L','G','GS','CG','SHO','SV','IPouts','H','ER','HR','BB','SO','IBB','WP','HBP','BK','BFP','GF','R','SH','SF','GIDP']].sum(); pit_year['IP']=pit_year['IPouts']/3; pit_year['ERA_calc']=safe_div(9*pit_year['ER'],pit_year['IP']); pit_year['WHIP']=safe_div(pit_year['BB']+pit_year['H'],pit_year['IP']); pit_year['K_BB']=safe_div(pit_year['SO'],pit_year['BB']); pit_year['K9']=safe_div(9*pit_year['SO'],pit_year['IP']); pit_year['BB9']=safe_div(9*pit_year['BB'],pit_year['IP']); pit_year['HR9']=safe_div(9*pit_year['HR'],pit_year['IP'])
    lp=pit_year.groupby('yearID').agg(lg_ER=('ER','sum'),lg_IP=('IP','sum')).reset_index(); lp['lg_ERA']=safe_div(9*lp['lg_ER'],lp['lg_IP']); sdera=pit_year[pit_year.IP>=30].groupby('yearID')['ERA_calc'].std().rename('lg_ERA_sd').reset_index(); pit_year=pit_year.merge(lp[['yearID','lg_ERA']],on='yearID',how='left').merge(sdera,on='yearID',how='left'); pit_year['ERA_plus_proxy']=100*pd.Series(pit_year['lg_ERA']).divide(pit_year['ERA_calc'].replace(0,np.nan)); pit_year['ERA_z_good']=pd.Series(pit_year['lg_ERA']-pit_year['ERA_calc']).divide(pit_year['lg_ERA_sd'].replace(0,np.nan))
    pit_c=pit_year.groupby('playerID')[['W','L','G','GS','CG','SHO','SV','IPouts','H','ER','HR','BB','SO','IBB','WP','HBP','BK','BFP','GF','R','IP']].sum().add_prefix('pit_').reset_index(); pit_c['pit_ERA']=safe_div(9*pit_c['pit_ER'],pit_c['pit_IP']); pit_c['pit_WHIP']=safe_div(pit_c['pit_BB']+pit_c['pit_H'],pit_c['pit_IP']); pit_c['pit_K_BB']=safe_div(pit_c['pit_SO'],pit_c['pit_BB']); pit_c['pit_K9']=safe_div(9*pit_c['pit_SO'],pit_c['pit_IP']); pit_c['pit_BB9']=safe_div(9*pit_c['pit_BB'],pit_c['pit_IP']); pit_c['pit_HR9']=safe_div(9*pit_c['pit_HR'],pit_c['pit_IP']); pit_c['pit_win_pct']=safe_div(pit_c['pit_W'],pit_c['pit_W']+pit_c['pit_L']); pit_c['pit_start_share']=safe_div(pit_c['pit_GS'],pit_c['pit_G']); pit_c['pit_complete_game_share']=safe_div(pit_c['pit_CG'],pit_c['pit_GS']); pit_c['pit_save_share']=safe_div(pit_c['pit_SV'],pit_c['pit_G'])
    era_pit=pit_year.groupby('playerID').agg(pit_mean_ERA_z_good=('ERA_z_good','mean'),pit_max_ERA_z_good=('ERA_z_good','max'),pit_mean_ERA_plus_proxy=('ERA_plus_proxy','mean'),pit_max_ERA_plus_proxy=('ERA_plus_proxy','max')).reset_index(); pit_c=pit_c.merge(era_pit,on='playerID',how='left')
    for df in [rolling_peak(pit_year,'playerID','yearID',['W','SO','SV','IP'],3,'pit','sum'),rolling_peak(pit_year,'playerID','yearID',['ERA_plus_proxy','ERA_z_good'],3,'pit','mean'),rolling_peak(pit_year,'playerID','yearID',['W','SO','SV','IP'],5,'pit','sum'),rolling_peak(pit_year,'playerID','yearID',['ERA_plus_proxy','ERA_z_good'],5,'pit','mean')]: pit_c=pit_c.merge(df,on='playerID',how='left')
    # Fielding / Appearances.
    fld=add_num(fielding.copy(), ['G','GS','InnOuts','PO','A','E','DP','PB','WP','SB','CS','ZR']); fld_sum=fld.groupby('playerID')[['G','GS','InnOuts','PO','A','E','DP']].sum().add_prefix('fld_').reset_index(); fld_sum['fld_fielding_pct']=safe_div(fld_sum['fld_PO']+fld_sum['fld_A'],fld_sum['fld_PO']+fld_sum['fld_A']+fld_sum['fld_E'])

    # Prefer the full Lahman Appearances table when available — it is the
    # canonical source for primary position because it is already aggregated to
    # player-year with a column per position. Falls back to Fielding.csv if the
    # Kaggle-only mirror is in use (where Appearances.csv was unavailable).
    if appearances is not None and len(appearances) > 0:
        pos_map = {'G_p':'P','G_c':'C','G_1b':'1B','G_2b':'2B','G_3b':'3B',
                   'G_ss':'SS','G_lf':'LF','G_cf':'CF','G_rf':'RF','G_of':'OF',
                   'G_dh':'DH','G_ph':'PH','G_pr':'PR'}
        app_cols = [c for c in pos_map if c in appearances.columns]
        app = add_num(appearances.copy(), app_cols)
        app_sum = app.groupby('playerID')[app_cols].sum().reset_index()
        long = app_sum.melt(id_vars='playerID', value_vars=app_cols,
                            var_name='app_position_col', value_name='G')
        long['POS'] = long['app_position_col'].map(pos_map)
        fld_pos = (long[long['G'] > 0]
                   .sort_values(['playerID', 'G'], ascending=[True, False]))
        primary = (fld_pos.drop_duplicates('playerID')
                   .rename(columns={'POS': 'primary_position',
                                    'G': 'primary_position_games'})
                   [['playerID', 'primary_position', 'primary_position_games']])
    else:
        fld_pos = (fld.groupby(['playerID', 'POS'])['G'].sum().reset_index()
                   .sort_values(['playerID', 'G'], ascending=[True, False]))
        primary = (fld_pos.drop_duplicates('playerID')
                   .rename(columns={'POS': 'primary_position',
                                    'G': 'primary_position_games'})
                   [['playerID', 'primary_position', 'primary_position_games']])
    fld_sum = fld_sum.merge(primary, on='playerID', how='left')
    fw=fld_pos.pivot_table(index='playerID',columns='POS',values='G',fill_value=0,aggfunc='sum'); fw.columns=[f'pos_games_{c}' for c in fw.columns]; fld_sum=fld_sum.merge(fw.reset_index(),on='playerID',how='left')
    fo=add_num(fielding_of.copy(),['Glf','Gcf','Grf']); fld_sum=fld_sum.merge(fo.groupby('playerID')[['Glf','Gcf','Grf']].sum().add_prefix('of_').reset_index(),on='playerID',how='left')
    # Awards/Allstar/Salary/Postseason
    aw=awards.copy(); aw['awardID_clean']=aw['awardID'].fillna('').str.lower(); award_agg=aw.groupby('playerID').agg(award_total=('awardID','count'),award_unique=('awardID','nunique'),award_years=('yearID','nunique')).reset_index()
    patterns={'mvp':'mvp|most valuable','cyyoung':'cy young','gold_glove':'gold glove','silver_slugger':'silver slugger','rookie':'rookie','triple_crown':'triple crown','world_series_mvp':'world series','all_star_mvp':'all-star|all star'}
    for key,pat in patterns.items(): award_agg=award_agg.merge(aw[aw['awardID_clean'].str.contains(pat,regex=True,na=False)].groupby('playerID').size().rename(f'award_{key}_count').reset_index(),on='playerID',how='left')
    asr=add_num(awardshare.copy(),['pointsWon','pointsMax','votesFirst']); asr['award_share_ratio']=safe_div(asr['pointsWon'],asr['pointsMax']); asr_agg=asr.groupby('playerID').agg(awardshare_rows=('awardID','count'),awardshare_pointsWon_sum=('pointsWon','sum'),awardshare_pointsMax_sum=('pointsMax','sum'),awardshare_votesFirst_sum=('votesFirst','sum'),awardshare_max_share=('award_share_ratio','max'),awardshare_mean_share=('award_share_ratio','mean')).reset_index(); award_agg=award_agg.merge(asr_agg,on='playerID',how='outer').fillna(0)
    als=add_num(allstar.copy(),['GP']); als['startingPos']=pd.to_numeric(als['startingPos'],errors='coerce'); allstar_agg=als.groupby('playerID').agg(allstar_games=('gameID','count'),allstar_years=('yearID','nunique'),allstar_appearances=('GP','sum'),allstar_starts=('startingPos',lambda x:x.notna().sum())).reset_index()
    sal=salaries.copy(); sal['salary']=pd.to_numeric(sal['salary'],errors='coerce'); sal_agg=sal.groupby('playerID').agg(salary_years=('yearID','nunique'),salary_total=('salary','sum'),salary_mean=('salary','mean'),salary_median=('salary','median'),salary_max=('salary','max'),salary_first_year=('yearID','min'),salary_last_year=('yearID','max')).reset_index()
    bp=add_num(batting_post.copy(),['G','AB','R','H','2B','3B','HR','RBI','SB','CS','BB','SO','IBB','HBP','SH','SF','GIDP']); bp['PA']=bp[['AB','BB','HBP','SH','SF']].sum(axis=1); bp['TB']=bp['H']+bp['2B']+2*bp['3B']+3*bp['HR']; bp_agg=bp.groupby('playerID')[['G','AB','R','H','2B','3B','HR','RBI','SB','CS','BB','SO','PA','TB']].sum().add_prefix('batpost_').reset_index(); bp_agg['batpost_OPS']=safe_div(bp_agg['batpost_H']+bp_agg['batpost_BB'],bp_agg['batpost_AB']+bp_agg['batpost_BB'])+safe_div(bp_agg['batpost_TB'],bp_agg['batpost_AB'])
    pp=add_num(pitching_post.copy(),['W','L','G','GS','CG','SHO','SV','IPouts','H','ER','HR','BB','SO','R']); pp['IP']=pp['IPouts']/3; pp_agg=pp.groupby('playerID')[['W','L','G','GS','CG','SHO','SV','IPouts','H','ER','HR','BB','SO','R','IP']].sum().add_prefix('pitpost_').reset_index(); pp_agg['pitpost_ERA']=safe_div(9*pp_agg['pitpost_ER'],pp_agg['pitpost_IP']); pp_agg['pitpost_WHIP']=safe_div(pp_agg['pitpost_BB']+pp_agg['pitpost_H'],pp_agg['pitpost_IP'])
    # target
    hofp=hof[hof['category'].fillna('')=='Player'].copy(); hofp['inducted_binary']=(hofp['inducted'].astype(str).str.upper()=='Y').astype(int); ind=hofp.groupby('playerID')['inducted_binary'].max().rename('inducted').reset_index(); first_ind=hofp[hofp.inducted_binary==1].groupby('playerID')['yearid'].min().rename('induction_year').reset_index(); ballots=hofp.groupby('playerID').agg(hof_ballot_rows=('yearid','count'),hof_first_ballot_year=('yearid','min'),hof_last_ballot_year=('yearid','max')).reset_index()
    # merge
    feat=base.copy()
    for df in [bat_c,pit_c,fld_sum,award_agg,allstar_agg,sal_agg,bp_agg,pp_agg,ind,first_ind,ballots]: feat=feat.merge(df,on='playerID',how='left')
    feat['inducted']=feat['inducted'].fillna(0).astype(int); max_hof_year=int(hofp['yearid'].max()); feat['is_eligible']=((feat['n_mlb_seasons']>=10)&(feat['eligibility_year']<=max_hof_year)).astype(int); feat['model_eligible_pool']=((feat['is_eligible']==1)|(feat['inducted']==1)).astype(int)
    feat['bat_PA']=feat['bat_PA'].fillna(0); feat['pit_IP']=feat['pit_IP'].fillna(0); feat['primary_role']=np.select([(feat['pit_IP']>200)&(feat['bat_PA']>1000),(feat['pit_IP']>100)&(feat['bat_PA']<1000),(feat['primary_position'].fillna('')=='P')],['Two-Way','Pitcher','Pitcher'],default='Hitter')
    bins=[0,1900,1920,1947,1961,1976,1993,2005,2017,3000]; labels=['19th century','Dead-ball','Live-ball/pre-integration','Integration','Expansion','Free-agency','Steroid','Post-steroid','Modern']; feat['debut_era']=pd.cut(feat['debut_year'],bins=bins,labels=labels,right=False).astype(str); feat['debut_decade']=(np.floor(feat['debut_year']/10)*10).astype('Int64').astype(str)
    for col in ['birthYear','birthMonth','weight','height','bats','throws','debut_year','final_year','bmi','debut_age','final_age','primary_position','salary_total']:
        if col in feat.columns: feat[f'{col}_missing']=feat[col].isna().astype(int)
    feat=feat.replace([np.inf,-np.inf],np.nan)

    # ===================================================================
    # Extended feature engineering: outlier flags, power transforms,
    # concentration / diversity metrics, and HoF-specific interaction terms.
    # Each block ties to a specific narrative beat in the HoF story.
    # ===================================================================
    feat = add_iqr_outlier_flags(
        feat,
        columns=["bat_HR", "bat_H", "bat_RBI",
                 "pit_W", "pit_SO", "pit_SV",
                 "bat_max_OPS_plus_proxy", "pit_max_ERA_plus_proxy"]
    )
    feat = add_yeo_johnson_transforms(
        feat,
        columns=["bat_HR", "bat_H", "bat_RBI",
                 "pit_W", "pit_SO", "pit_SV"]
    )
    feat = add_concentration_metrics(feat)
    feat = add_interaction_terms(feat)

    if processed_dir is not None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        feat.to_csv(processed_dir / 'player_features_base.csv', index=False)
    return feat


# ======================================================================
# Helper feature constructors used by build_player_features
# ======================================================================
def add_iqr_outlier_flags(df: pd.DataFrame, columns: list, k: float = 1.5) -> pd.DataFrame:
    """Flag IQR-based outliers WITHOUT capping them.

    We use [Q1 - 1.5*IQR, Q3 + 1.5*IQR] for outlier detection on
    weather and complaint variables. For HoF career stats the flagged
    "outliers" are precisely the legends — Aaron's 755 HR, Cy Young's 511
    wins, Walter Johnson's 110 shutouts. These are the SIGNAL the HoF model
    must learn, not noise. We document them but never cap them.

    Storytelling: "the players the model needs to find ARE the IQR outliers."
    """
    df = df.copy()
    seen = set()
    for c in columns:
        if c is None or c in seen or c not in df.columns:
            continue
        seen.add(c)
        s = df[c].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 20:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lb, ub = q1 - k * iqr, q3 + k * iqr
        df[f"{c}_outlier_high"] = (df[c] > ub).astype("Int64")
        df[f"{c}_outlier_low"] = (df[c] < lb).astype("Int64")
    # Aggregate IQR-outlier flag count: how many career stats put a player
    # in the legendary tail? (More = stronger HoF candidate, by construction.)
    flag_cols = [c for c in df.columns if c.endswith("_outlier_high")]
    if flag_cols:
        df["n_iqr_outlier_high"] = df[flag_cols].sum(axis=1, skipna=True).astype("Int64")
    return df


def add_yeo_johnson_transforms(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Yeo-Johnson power transforms on heavy-tailed counting stats.

    We apply the Yeo-Johnson variant (handles zeros / negatives gracefully) to career
    counting stats which are similarly right-skewed. This stabilizes
    variance for any linear/regularized models that prefer approximate
    normality.
    """
    try:
        from sklearn.preprocessing import PowerTransformer
    except ImportError:
        return df
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].replace([np.inf, -np.inf], np.nan)
        if s.dropna().shape[0] < 10:
            continue
        filled = s.fillna(s.median()).values.reshape(-1, 1)
        try:
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            df[f"{col}_yj"] = pt.fit_transform(filled).ravel()
        except Exception:
            continue
    return df


def add_concentration_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Career-profile concentration (HHI) and diversity (entropy).

    We measure entropy and HHI to characterize each player's career profile:

      * award_entropy / award_hhi : did this player win MANY award types
        (5-tool legend) or just one (specialist)?
      * position_versatility / position_concentration : did this player play
        one position (specialist) or many (utility / two-way)?

    Storytelling: "5-tool legends" and "single-skill specialists" are both
    HoF archetypes; the model needs features that distinguish them.
    """
    df = df.copy()

    # Award diversity. Award count columns are named `award_*_count`
    # (e.g., award_mvp_count, award_cyyoung_count, ...).
    award_count_cols = [c for c in df.columns
                         if c.startswith("award_") and c.endswith("_count")
                         and not c.endswith("_yj")
                         and pd.api.types.is_numeric_dtype(df[c])]
    aw_cols = award_count_cols
    if aw_cols:
        award_total = df[aw_cols].sum(axis=1, skipna=True)
        share_df = df[aw_cols].div(award_total.replace(0, np.nan), axis=0).fillna(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["award_entropy"] = -(share_df * np.log(share_df + 1e-12)).sum(axis=1)
            df["award_hhi"] = (share_df ** 2).sum(axis=1)
        df.loc[award_total == 0, ["award_entropy", "award_hhi"]] = np.nan
        df["career_award_total_norm"] = award_total

    # Position versatility from one-hot pos_games_*
    pos_cols = [c for c in df.columns if c.startswith("pos_games_")
                and pd.api.types.is_numeric_dtype(df[c])]
    if pos_cols:
        pos_total = df[pos_cols].sum(axis=1, skipna=True)
        share_df = df[pos_cols].div(pos_total.replace(0, np.nan), axis=0).fillna(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["position_versatility_entropy"] = (
                -(share_df * np.log(share_df + 1e-12)).sum(axis=1))
            df["position_concentration_hhi"] = (share_df ** 2).sum(axis=1)
        df.loc[pos_total == 0,
               ["position_versatility_entropy", "position_concentration_hhi"]] = np.nan

    return df


def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Baseball-specific interaction terms for HoF prediction.

    These interaction terms capture HoF-defining compound signals:

      * peak_x_longevity      — sustained excellence (high peak with long career)
      * allstar_x_pos_scarcity — rare-position superstar (e.g., HoF catcher)
      * mvp_x_postseason_HR   — big-stage performer
      * is_pitcher_x_K        — pitcher-specific signal
      * steroid_era_x_HR      — captures the historical "steroid penalty"
                                in voter behavior

    Storytelling: "HoF voters reward sustained excellence at scarce positions
    in big moments; they penalize tainted-era totals."
    """
    df = df.copy()

    # Sustained excellence: high era-adjusted peak × long career
    if {"bat_max_OPS_plus_proxy", "n_mlb_seasons"}.issubset(df.columns):
        df["peak_x_longevity"] = (
            df["bat_max_OPS_plus_proxy"].fillna(0) * df["n_mlb_seasons"].fillna(0))

    # Position scarcity: catchers / shortstops are scarcer in HoF
    if "primary_position" in df.columns:
        scarcity_map = {"C": 5, "SS": 5, "2B": 4, "3B": 4, "CF": 4,
                        "OF": 3, "LF": 3, "RF": 3, "1B": 2, "DH": 1, "P": 0}
        df["primary_pos_scarcity"] = (
            df["primary_position"].map(scarcity_map).fillna(2).astype(float))

    if {"allstar_games", "primary_pos_scarcity"}.issubset(df.columns):
        df["allstar_x_pos_scarcity"] = (
            df["allstar_games"].fillna(0) * df["primary_pos_scarcity"].fillna(0))

    # Big-stage performer: MVPs × postseason HR  (or regular-season HR fallback)
    if {"award_mvp_count", "batpost_HR"}.issubset(df.columns):
        df["mvp_x_postseason_HR"] = (
            df["award_mvp_count"].fillna(0) * df["batpost_HR"].fillna(0))
    elif {"award_mvp_count", "bat_HR"}.issubset(df.columns):
        df["mvp_x_career_HR"] = (
            df["award_mvp_count"].fillna(0) * df["bat_HR"].fillna(0))

    # Pitcher-specific signal
    if {"primary_role", "pit_SO"}.issubset(df.columns):
        is_p = (df["primary_role"] == "Pitcher").astype(float)
        df["is_pitcher_x_K"] = is_p * df["pit_SO"].fillna(0)

    # Steroid-era penalty proxy. Captures the voter-behavior bias for Bonds,
    # Clemens, A-Rod. The model can learn to discount these totals.
    if {"debut_era", "bat_HR"}.issubset(df.columns):
        is_steroid = (df["debut_era"] == "Steroid").astype(float)
        df["steroid_era_x_HR"] = is_steroid * df["bat_HR"].fillna(0)

    return df

if __name__ == '__main__':
    from src.config import DATA_RAW_KAGGLE, DATA_PROCESSED
    df = build_player_features(DATA_RAW_KAGGLE, DATA_PROCESSED)
    print(f'Built player_features_base.csv with {df.shape[0]} rows and {df.shape[1]} columns')
    print(f"Eligible modeling pool: {int(df['model_eligible_pool'].sum())}; positives: {int(df.loc[df['model_eligible_pool']==1,'inducted'].sum())}")
