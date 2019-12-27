drop table if exists sdk_user.hejy_temp_1;
create table sdk_user.hejy_temp_1 as
select dhid,app_3,app_10,app_12,app_14,app_15
from sdk_user.hejy_glmnet_model_22001946_20181217_1
limit 50;


drop table if exists sdk_user.hejy_temp_2;
create table sdk_user.hejy_temp_2 stored as orc as
select dhid , 1/(1+exp(-( 0 + sum(pred)))) AS pred 
from (
select dhid,
case when app_3<3 then 
(case when app_10<13.6458502 then -0.006400852
when app_10>=13.6458502 then 
(case when app_12<42.5 then 0.033544872
when app_12>=42.5 then 0.00980747491
when app_12 is null then 0.00980747491
end)
when app_10 is null then -0.006400852
end)
when app_3>=3 then 
(case when app_12<6.5 then 
(case when app_14<3 then 0.0521093272
when app_14>=3 then -0.0224915836
when app_14 is null then 0.0521093272
end)
when app_12>=6.5 then -0.0162228346
when app_12 is null then -0.0162228346
end)
when app_3 is null then 
(case when app_10<13.6458502 then -0.006400852
when app_10>=13.6458502 then 
(case when app_12<42.5 then 0.033544872
when app_12>=42.5 then 0.00980747491
when app_12 is null then 0.00980747491
end)
when app_10 is null then -0.006400852
end)end as pred
from sdk_user.hejy_temp_1
union all
select dhid,
case when app_3<3 then 
(case when app_10<52.2770996 then 
(case when app_15<1 then 0.0001473382
when app_15>=1 then 0.0348955654
when app_15 is null then 0.0001473382
end)
when app_10>=52.2770996 then 0.0321655758
when app_10 is null then 
(case when app_15<1 then 0.0001473382
when app_15>=1 then 0.0348955654
when app_15 is null then 0.0001473382
end)
end)
when app_3>=3 then 
(case when app_12<6.5 then 
(case when app_14<3 then 0.0472049229
when app_14>=3 then -0.0203923695
when app_14 is null then 0.0472049229
end)
when app_12>=6.5 then -0.0146040637
when app_12 is null then -0.0146040637
end)
when app_3 is null then 
(case when app_10<52.2770996 then 
(case when app_15<1 then 0.0001473382
when app_15>=1 then 0.0348955654
when app_15 is null then 0.0001473382
end)
when app_10>=52.2770996 then 0.0321655758
when app_10 is null then 
(case when app_15<1 then 0.0001473382
when app_15>=1 then 0.0348955654
when app_15 is null then 0.0001473382
end)
end)end as pred
from sdk_user.hejy_temp_1
union all
select dhid,
case when app_3<3 then 
(case when app_10<13.6458502 then -0.00585523667
when app_10>=13.6458502 then 0.0201724116
when app_10 is null then -0.00585523667
end)
when app_3>=3 then -0.0114313215
when app_3 is null then 
(case when app_10<13.6458502 then -0.00585523667
when app_10>=13.6458502 then 0.0201724116
when app_10 is null then -0.00585523667
end)end as pred
from sdk_user.hejy_temp_1
)a
group by dhid;

