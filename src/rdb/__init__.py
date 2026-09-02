"""RDB 계층 — pool(커넥션 풀) + 테이블별 repo. raw SQL 은 이 패키지 안에만 존재한다.

도메인 무관 공통 테이블(t_video 등)과 compose 서비스 소유 테이블(t_compose 등)만 여기 둔다 —
도메인 접미사 테이블(*_baseball)의 repo 는 domains/*/repo/ 소유.
"""

