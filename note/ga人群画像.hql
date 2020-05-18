drop table if exists test.hejy_temp_ga_profile_1;
create table test.hejy_temp_ga_profile_1 as
select phone,device[0] as device
from test.wuzc_phone_4ma_20200507ynsr_tsk1;

drop table if exists test.hejy_temp_ga_profile_1_1;
create table test.hejy_temp_ga_profile_1_1 as
select a.phone,b.* 
from test.hejy_temp_ga_profile_1 a
inner join 
(select *
from rp_mobdi_app.device_profile_label_full_par
where version='20200511.1001')b
on a.device=b.device;

drop table test.hejy_20200509_xb_profile_clean;
create table test.hejy_20200509_xb_profile_clean as
select '1' as group_id, t2.device
    , t2.gender, t2.agebin, t2.edu, t2.income, t2.married
    , t2.kids, t2.occupation, t2.industry, t2.car, t2.house
    , case
        when t2.province_cn in (
            '广东','江苏','浙江','山东','四川','河南',
            '湖北','湖南','河北','福建','安徽','上海','辽宁',
            '广西','江西','北京','重庆','云南','陕西','黑龙江',
            '吉林','贵州','山西','内蒙古','天津','新疆','甘肃',
            '海南','宁夏','青海','香港','西藏','台湾','澳门')
            then t2.province_cn
        else '未知'
    end as province_cn
    , t2.city_cn, t2.city_level, t2.city_level_1001
    , t2.model_level, t2.cell_factory, t2.carrier, t2.model, t2.price, t2.sysver, t2.screensize, t2.network
    , case
        when t2.price < 1000 then '1000以下'
        when t2.price < 2000 then '1000-1999'
        when t2.price < 3000 then '2000-2999'
        when t2.price < 4000 then '3000-3999'
        when t2.price >= 4000 then '4000以上'
        else '未知'
    end as price_level
    , t2.identity -- 特殊身份
    , t2.special_time -- 特定时期
    , t2.life_stage -- 人生阶段
    , t2.segment -- 人群
    , t2.group_list -- 人群细分
    , t2.applist
    , t2.last_active
    , substring(t2.last_active, 1, 6) as last_active_month
from test.hejy_temp_ga_profile_1_1 t2
where t2.country = 'cn';


-- drop table test.wanghz_global_profile;
-- create table test.wanghz_global_profile as
-- select 'global' as group_id, t2.device
--     , t2.gender, t2.agebin, t2.edu, t2.income, t2.married
--     , t2.kids, t2.occupation, t2.industry, t2.car, t2.house
--     , case
--         when t2.province_cn in (
--             '广东','江苏','浙江','山东','四川','河南',
--             '湖北','湖南','河北','福建','安徽','上海','辽宁',
--             '广西','江西','北京','重庆','云南','陕西','黑龙江',
--             '吉林','贵州','山西','内蒙古','天津','新疆','甘肃',
--             '海南','宁夏','青海','香港','西藏','台湾','澳门')
--             then t2.province_cn
--         else '未知'
--     end as province_cn
--     , t2.city_cn, t2.city_level, t2.city_level_1001
--     , t2.model_level, t2.cell_factory, t2.carrier, t2.model, t2.price, t2.sysver, t2.screensize, t2.network
--     , case
--         when t2.price < 1000 then '1000以下'
--         when t2.price < 2000 then '1000-1999'
--         when t2.price < 3000 then '2000-2999'
--         when t2.price < 4000 then '3000-3999'
--         when t2.price >= 4000 then '4000以上'
--         else '未知'
--     end as price_level
--     , t2.identity -- 特殊身份
--     , t2.special_time -- 特定时期
--     , t2.life_stage -- 人生阶段
--     , t2.segment -- 人群
--     , t2.group_list -- 人群细分
--     , t2.applist
--     , t2.last_active
--     , substring(t2.last_active, 1, 6) as last_active_month
-- from rp_mobdi_app.rp_device_profile_full_view t2
-- where t2.country = 'cn' and rand()<0.02 ;
--52140794


spark2-shell \
--name "" \
--jars /home/hejy/dir_jar/tools.jar \
--driver-memory 15g \
--executor-memory 12g \
--executor-cores 3 \
--conf spark.shuffle.service.enabled=true \
--conf spark.dynamicAllocation.enabled=true \
--conf spark.dynamicAllocation.minExecutors=20 \
--conf spark.dynamicAllocation.maxExecutors=60

import profile.Profile
val p = new Profile()
p.featureDistribution(
  "test.hejy_20200509_xb_profile_clean",//种子表
  "test.hejy_20200210_ly_seed_profile_clean_dist",//落地表
  maxRank=Map("carrier" -> 3, "model" -> 100, "cell_factory" -> 100, "sysver" -> 100, "screensize" -> 100),
  profileMappingTable="test.zhanjf_profile_mapping",//画像映射表
  globalProfileTable="test.wanghz_global_profile")//全局表


p.appDistribution(
  "test.hejy_20200509_xb_profile_clean",//种子表
  Array(
    "test.hejy_20200509_xb_profile_clean_res1",//
    "test.hejy_20200509_xb_profile_clean_res2",//
    "test.hejy_20200509_xb_profile_clean_res3"//
  ),
  Array(
    Array("cate_l1"),
    Array("cate_l1", "cate_l2"),
    Array("cate_l1", "cate_l2", "appname")
  ),
  joinOnCol="apppkg", idCol="device", applistCol="applist", groupCol="group_id",
  cateMappingTable="",
  globalTable="test.wanghz_global_profile"//全局表
)
-- test.hejy_20200509_xb_profile_clean_res1
-- test.hejy_20200509_xb_profile_clean_res2
-- test.hejy_20200509_xb_profile_clean_res3
-- test.hejy_20200210_ly_seed_profile_clean_dist

SELECT 1 as flag,cate_l1 as cate,distribution,tgi from test.hejy_20200509_xb_profile_clean_res1
union all 
SELECT 2 as flag,cate_l2 as cate,distribution,tgi from test.hejy_20200509_xb_profile_clean_res2
union all 
SELECT 3 as flag,appname as cate,distribution,tgi from 
(select * from test.hejy_20200509_xb_profile_clean_res3
order by appname desc 
limit 50)a;

select * from test.hejy_20200210_ly_seed_profile_clean_dist limit 100;



-- 活跃列表
drop table test.hejy_temp_ga_profile_4;
create table test.hejy_temp_ga_profile_4 as 
select a.*,b.appname,b.cate_l1,b.cate_l2 from 
(
select * 
from rp_mobdi_app.app_active_monthly
where month='202004'
)a 
left join 
(
select apppkg, max(appname) appname, max(cate_l1) cate_l1, max(cate_l2) cate_l2
from dm_sdk_mapping.app_category_mapping_par 
where version='1000' 
group by apppkg
)b 
on a.apppkg=b.apppkg;

drop table test.hejy_temp_ga_profile_4_1;
create table test.hejy_temp_ga_profile_4_1 as 
select b.* from test.hejy_temp_ga_profile_1 a 
inner join test.hejy_temp_ga_profile_4 b 
on a.device=b.device;

drop table test.hejy_temp_ga_profile_4_2;
create table test.hejy_temp_ga_profile_4_2 as 
select a.cate_l1,num1/5 as rate1,num2/80245586 as rate2,(num1/5)/(num2/80245586) as tgi from
(select cate_l1,count(1) as num1 from 
    (select device,cate_l1 
    from test.hejy_temp_ga_profile_4_1
    group by device,cate_l1)t1
group by cate_l1)a
left outer join 
(select cate_l1,count(1) as num2 from 
    (select device,cate_l1 
    from test.hejy_temp_ga_profile_4
    group by device,cate_l1)t1
group by cate_l1)b 
on a.cate_l1=b.cate_l1;

drop table test.hejy_temp_ga_profile_4_2_1;
create table test.hejy_temp_ga_profile_4_2_1 as 
select a.cate_l1,num1/30 as rate1,num2/30 as rate2,(num1/30)/(num2/30) as tgi from
(select cate_l1,avg(days) as num1
from test.hejy_temp_ga_profile_4_1
group by cate_l1)a
left outer join 
(select cate_l1,avg(days) as num2
from test.hejy_temp_ga_profile_4
group by cate_l1)b 
on a.cate_l1=b.cate_l1;

drop table test.hejy_temp_ga_profile_4_3;
create table test.hejy_temp_ga_profile_4_3 as 
select a.cate_l2,num1/5 as rate1,num2/80245586 as rate2,(num1/5)/(num2/80245586) as tgi from
(select cate_l2,count(1) as num1 from 
    (select device,cate_l2 
    from test.hejy_temp_ga_profile_4_1
    group by device,cate_l2)t1
group by cate_l2)a
left outer join 
(select cate_l2,count(1) as num2 from 
    (select device,cate_l2 
    from test.hejy_temp_ga_profile_4
    group by device,cate_l2)t1
group by cate_l2)b 
on a.cate_l2=b.cate_l2;

drop table test.hejy_temp_ga_profile_4_3_1;
create table test.hejy_temp_ga_profile_4_3_1 as 
select a.cate_l2,num1/30 as rate1,num2/30 as rate2,(num1/30)/(num2/30) as tgi from
(select cate_l2,avg(days) as num1
from test.hejy_temp_ga_profile_4_1
group by cate_l2)a
left outer join 
(select cate_l2,avg(days) as num2
from test.hejy_temp_ga_profile_4
group by cate_l2)b 
on a.cate_l2=b.cate_l2;

--计算TGI
-- drop table test.hejy_temp_ga_profile_4_4;
-- create table  test.hejy_temp_ga_profile_4_4 as 
-- select x.apppkg, x.appname, x.cate_l1, x.cate_l2, x.cnt1, y.cnt3, x.cnt1/5 as ratio_target, y.cnt3/80245586 as ratio_all, (x.cnt1/5)/(y.cnt3/80245586)*100 as TGI
-- from 
-- (
--     select apppkg, appname, cate_l1, cate_l2, count(*) as cnt1 
--     from 
--     (
--         select device, apppkg, appname, cate_l1, cate_l2 
--         from test.hejy_temp_ga_profile_4
--         group by device, apppkg, appname, cate_l1, cate_l2
--     )a
--     group by apppkg, appname, cate_l1, cate_l2
-- )x
-- join 
-- (
--     select apppkg, count(*) as cnt3
--     from 
--     (
--         select device, apppkg 
--         from rp_mobdi_app.app_active_monthly 
--         where month='202004'
--         group by device,apppkg
--     )b
--     group by apppkg
-- )y 
-- on x.apppkg=y.apppkg;


-- lvyou
drop table test.hejy_temp_ga_profile_2;
create table test.hejy_temp_ga_profile_2 as
select b.*
from 
(
    select device
    from test.hejy_temp_ga_profile_1 
    group by device 
)a 
join 
(
    select * from rp_mobdi_app.travel_label_monthly
    where day='20200501'
)
b 
on a.device=b.device;

select * from test.hejy_temp_ga_profile_2 limit 100;

drop table test.hejy_temp_ga_profile_2_1;
create table test.hejy_temp_ga_profile_2_1 as
select device, 'country' as feature, country1 as value_name
from 
(
    select device, country
    from test.hejy_temp_ga_profile_2
    where country<>''
)a lateral view explode(split(country,',')) a as country1
union ALL
select device,'travel_type' as feature, travel_type1 as value_name
from 
(
    select device, travel_type
    from test.hejy_temp_ga_profile_2
    where travel_type<>'unknown' and travel_type<>''
)a lateral view explode(split(travel_type,',')) a as travel_type1
union ALL
select device,'vaca_flag' as feature, vaca_flag1 as value_name
from 
(
    select device, vaca_flag
    from test.hejy_temp_ga_profile_2
    where vaca_flag<>'unknown' and vaca_flag<>''
    )a lateral view explode(split(vaca_flag,',')) a as vaca_flag1
union ALL
select device,'travel_time' as feature,  travel_time1 as value_name
from 
(
    select device,  travel_time
    from test.hejy_temp_ga_profile_2
    where travel_time<>'unknown' and travel_time<>''
)a lateral view explode(split(travel_time,',')) a as travel_time1
union ALL
select device,'traffic' as feature,  traffic1 as value_name
from 
(
    select device,  traffic
    from test.hejy_temp_ga_profile_2
    where traffic<>'unknown' and traffic<>''
)a lateral view explode(split(traffic,',')) a as traffic1
union ALL
select device,'travel_area' as feature,  travel_area1 as value_name
from 
(
    select device,  travel_area
    from test.hejy_temp_ga_profile_2
    where travel_area<>'unknown' and travel_area<>''
)a lateral view explode(split(travel_area,',')) a as travel_area1
union ALL
select device,'travel_channel' as feature,  travel_channel1 as value_name
from 
(
    select device,  travel_channel
    from test.hejy_temp_ga_profile_2
    where travel_channel<>'unknown' and travel_channel<>''
)a lateral view explode(split(travel_channel,',')) a as travel_channel1
;

select * from test.hejy_temp_ga_profile_2_1 limit 100;

drop table test.hejy_temp_ga_profile_2_2;
create table test.hejy_temp_ga_profile_2_2 as
select x.feature, x.value_name, count, total_count,count/total_count as distribution
from 
(
    select feature, value_name, count(distinct device) as count
    from test.hejy_temp_ga_profile_2_1
    group by feature, value_name
)x 
join 
(
    select feature, count(distinct(device)) as total_count
    from test.hejy_temp_ga_profile_2_1
    group by feature
)b 
on x.feature=b.feature;

select * from test.hejy_temp_ga_profile_2_2 limit 100;


-- canying
drop table test.hejy_temp_ga_profile_3;
create table test.hejy_temp_ga_profile_3 as
select b.*
from 
(
    select device
    from test.hejy_temp_ga_profile_1 
    group by device 
)a 
join 
(
    select * 
    from rp_mobdi_app.timewindow_offline_profile_v2
    where day='20200501' and timewindow='30' and flag in (6,9)
)b 
on a.device=b.device;

drop table test.hejy_temp_ga_profile_3_1;
create table test.hejy_temp_ga_profile_3_1 as
select *,count/total_count as distribution from
(select feature, value,count(distinct device) as count,56 as total_count
from 
(
    select device, feature, t.col1 as value, t.col2 as cnt
    from test.hejy_temp_ga_profile_3
    lateral view explode_tags(cnt) t as col1, col2
)a 
group by feature, value)c;

select * from test.hejy_temp_ga_profile_3_1 limit 100;

-- select count(1) from 
-- (select device 
-- from test.hejy_temp_ga_profile_3
-- group by device)a;
-- 56

drop table if exists test.hejy_temp_1;
create table test.hejy_temp_1 as
select device
from rp_mobdi_app.device_profile_label_full_par
where version='20200511.1001'
order by rand()
limit 100000;

drop table if exists test.hejy_temp_2;
create table test.hejy_temp_2 as
select 1 as flag,device 
from test.hejy_temp_ga_profile_1
union all 
select 0 as flag,device 
from test.hejy_temp_1;

drop table if exists test.hejy_temp_ga_profile_5;
create table test.hejy_temp_ga_profile_5 as
select a.*,b.hours,c.days
from test.hejy_temp_2 a
left outer join 
(select day,deviceid,sum(runtimes)/3600000 as hours
from dw_sdk_log.back_info
where day>'20200412'
and day<='20200512'
group by day,deviceid
having sum(runtimes)<=86400000)b 
on a.device=b.deviceid
left outer join 
(select device,count(1) as days from
(select day,device
from dm_mobdi_master.device_ip_info
where day>'20200412'
and day<='20200512'
group by day,device)t
group by device)c
on a.device=c.device;

select flag,avg(hours),avg(days) from test.hejy_temp_ga_profile_5 group by flag;
0	1.6131195046882902	15.893309887952892
1	2.836761156323876	15.011506276150628

drop table if exists test.hejy_temp_ga_profile_6;
create table test.hejy_temp_ga_profile_6 as
select a.flag,b.* 
from test.hejy_temp_2 a
inner join 
(select deviceid,clienttime,longitude,latitude
from dw_sdk_log.location_info
where day>'20200412'
and day<='20200512'
and location_type = '1'
and longitude > 73 
and longitude < 136 
and latitude > 3 
and latitude < 54
and (latitude not in ('39.915','30.0','43.856087','31.24916','25.895966','39.90719','45.825054','32.72628','39.90403','32.726276')
or longitude not in ('116.404','104.0','87.28649','121.4879','119.385025','116.391075','126.517975','120.32281','116.407524','120.32266'))
and deviceid rlike '^[0-9a-f]{40}$')b 
on a.device=b.deviceid;

select flag,count(1) from (select flag,device from test.hejy_temp_ga_profile_8 group by flag,device)a group by flag;

drop table if exists test.hejy_temp_ga_profile_7;
create table test.hejy_temp_ga_profile_7 as
select a.flag,b.* 
from test.hejy_temp_2 a
inner join 
(select device,datetime as clienttime,bssid
from dw_sdk_log.log_wifi_info
where day>'20200412'
and day<='20200512'
and networktype = 'wifi'
and trim(bssid) not in ('00:00:00:00:00:00', '02:00:00:00:00:00', 'ff:ff:ff:ff:ff:ff')
and trim(bssid) is not null
and regexp_replace(trim(lower(bssid)), '-|:|\\.|\073', '') rlike '^[0-9a-f]{12}$')b 
on a.device=b.device;

select flag,count(1) from (select flag,device from test.hejy_temp_ga_profile_7 group by flag,device)a group by flag;
16
9031

drop table if exists test.hejy_temp_ga_profile_8;
create table test.hejy_temp_ga_profile_8 as
select a.flag,b.* 
from test.hejy_temp_2 a
inner join 
(select *
from rp_mobdi_app.rp_device_sns_full)b 
on a.device=b.device;

select flag,count(1) from  test.hejy_temp_2 group by flag;
-- 100000
-- 582
select flag,count(1) from  test.hejy_temp_ga_profile_8 group by flag;
195
31452

drop table if exists test.hejy_temp_3;
create table test.hejy_temp_3 as
select * from 
(select split(phone,',')[0] as phone 
from dm_mobdi_mapping.android_id_mapping_full
where version='20200512.1001')a
order by rand()
limit 100000;

drop table if exists test.hejy_temp_4;
create table test.hejy_temp_4 as
select 1 as flag,phone
from test.hejy_temp_ga_profile_1
union all 
select 0 as flag,phone
from test.hejy_temp_3;

drop table if exists test.hejy_temp_ga_profile_9;
create table test.hejy_temp_ga_profile_9 as
select a.flag,b.* 
from test.hejy_temp_4 a
inner join 
(select * 
from rp_mobdi_app.phone_onedegree_rel
where day='20200508')b 
on a.phone=b.phone;


-- 18126
-- 356

582
100000

select flag,count(1) from  test.hejy_temp_ga_profile_9 group by flag;

-- drop table if exists test.hejy_temp_ga_profile_7;
-- create table test.hejy_temp_ga_profile_7 as
-- select b.* 
-- from test.hejy_temp_ga_profile_1 a
-- inner join 
-- (select device,applist
-- from rp_mobdi_app.device_profile_label_full_par
-- where version='20200511.1001')b
-- on a.device=b.device;

-- select * from test.hejy_temp_ga_profile_7 limit 100;

-- drop table if exists test.hejy_temp_ga_profile_7_1;
-- create table test.hejy_temp_ga_profile_7_1 as
-- select * from test.hejy_temp_ga_profile_7
-- LATERAL view explode(split(applist,','))t as app;

-- select * from test.hejy_temp_ga_profile_7_1 limit 100;


-- --渗透率和tgi计算
-- drop table if exists test.hejy_temp_ga_profile_7_2;
-- create table test.hejy_temp_ga_profile_7_2 as
-- select distinct app_info.appname,app_info.cate_l1,a.app,
-- a.num/b.all_num as radio,
-- (a.num/b.all_num)/(c.num/d.all_num) as tgi
-- from
--     (select app,count(distinct device) as num
--     from test.hejy_temp_ga_profile_7_1
--     where app is not null
--     and app <> ''
--     group by app
--     ) as a
-- left join
--     (select count(distinct device) as all_num
--     from test.hejy_temp_ga_profile_7_1
--     ) as b
-- left join
-- (select app,count(distinct device) as num
-- from rp_mobdi_app.rp_device_profile_full_view
-- LATERAL VIEW explode(split(applist,',')) c AS app
-- where app is not null
-- and app <> ''
-- group by app
-- ) as c
-- on a.app = c.app
-- left join
-- (
-- select
-- count(distinct device) as all_num
-- from rp_mobdi_app.rp_device_profile_full_view
-- LATERAL VIEW explode(split(applist,',')) c AS app
-- where app is not null
-- and app <> ''
-- ) as d
-- left join
-- dm_sdk_mapping.app_category_mapping_par as app_info
-- on app_info.version = '1000'
-- and a.app = app_info.apppkg
-- ;


-- select phone from test.wuzc_phone_4ma_20200507ynsr_tsk1


-- drop table test.wanghz_20190823_tag_name;
-- create table test.wanghz_20190823_tag_name as 
-- select tag, count(idfa) as cnt, count(idfa)/65271714 as per
-- from 
-- (
--     select a.idfa, a.tag_id, b.tag, a.weight
--     from 
--     (
--         select idfa, t.col1 as tag_id, t.col2 as weight  
--         from 
--         (
--             select *
--             from rp_mobdi_app.ios_active_tag_list
--             where day = '20190801'
--         )m lateral view explode_tags(tag_list) t as col1, col2
--     )a 
--     left join 
--     (
--         select *
--         from dm_sdk_mapping. tag_id_mapping_par 
--         where version = '1000'
--     )b 
--     on a.tag_id = b.id
--     group by a.idfa, a.tag_id, b.tag, a.weight
-- )x 
-- group by tag;
-- --657--

-- drop table test.wanghz_20190823_sanguo_tag_tgi;
-- create table test.wanghz_20190823_sanguo_tag_tgi as 
-- select a.tag, a.cnt1, a.per1, b.cnt, b.per, 100*(a.per1/b.per) as tgi
-- from 
-- (
--     select tag, count(distinct idfa) as cnt1, count(distinct idfa)/284440 as per1
--     from test.wanghz_20190823_sanguo_tag_name 
--     group by tag
-- )a 
-- join test.wanghz_20190823_tag_name b 
-- on a.tag = b.tag;
-- --593--