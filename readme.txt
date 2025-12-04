필요데이터
contents.csv
av_contents.csv
game_contents.csv
webnovel_contents.csv(필수x)
raw_item.csv
user_preferred_genres.csv(지금 유저에 대한 정보는 이거 하나만 전달.)


실행순서: make_bigraph(이분그래프 생성 및 필요한 데이터 생성) ->
make_itemgraph(아이템간의 그래프 생성)->make_itemembedding(아이템 임베딩 생성)
->make_userembedding(유저 데이터와 기존의 아이템 임베딩을 바탕으로 유저 임베딩 생성)->
rec_user_item_tok_k(유저에 대해 적절한 랭킹 탑 5 추천)->rec_user_item_domain_tok_k(도메인 써주는 추천)


