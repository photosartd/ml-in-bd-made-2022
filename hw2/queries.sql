select artist_mb from artists
where scrobbles_lastfm in 
(select max(scrobbles_lastfm) from artists);

select tag, count(1) as cnt from
(select explode(split(tags_mb, '; ')) as tag from artists) a
where tag is not null and tag != ""
group by tag
order by cnt desc
limit 1;

select * from 
(select tag, artist_mb from 
(select tag, artist_mb, row_number() over (partition by tag order by scrobbles_lastfm desc) as rn from
(select tag, artist_mb, avg(scrobbles_lastfm) as scrobbles_lastfm from 
(select tag_1 as tag, artist_mb, scrobbles_lastfm from 
(select split(tags_mb, '; ') as tag, artist_mb, scrobbles_lastfm from artists) d
lateral view explode(tag) ddd as tag_1
) dd
group by tag, artist_mb) e
) f
where rn = 1) best_artists
inner join
(select tag from 
(select tag, row_number() over (order by cnt desc) as rn from
(select tag, count(1) as cnt from 
(select explode(split(tags_mb, '; ')) as tag from artists) a
where tag != ''
group by tag
order by cnt desc) b) c
where rn <= 10) best_tags
on best_artists.tag = best_tags.tag;

select country_mb, artist_mb from 
(select country_mb, artist_mb, 
row_number() over (partition by country_mb order by scrobbles_lastfm desc) as rn
from 
(select country_mb, artist_mb, sum(scrobbles_lastfm) as scrobbles_lastfm from artists
group by country_mb, artist_mb) a
) b
where rn = 1;

