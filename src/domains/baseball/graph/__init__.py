"""야구 편성 LangGraph — 상태·노드·배선을 이 패키지가 소유한다.

구성 규약:
- 노드 1개 = 모듈 1개. 그 노드의 프롬프트·파서·구현이 한 파일에 있다 —
  노드 하나를 보려면 파일 하나만 열면 된다.
- 각 노드 모듈은 make_node(자원…) 팩토리로 노드 함수를 내어준다 —
  자원(repo·LLM)은 build.py 가 만들어 주입한다.
- 배선(노드 등록·엣지·분기)은 build.py 한 곳에만 있다.
- 진입은 domains/baseball/flow.py 의 run() — pipeline.dispatch 가 부른다.
"""
