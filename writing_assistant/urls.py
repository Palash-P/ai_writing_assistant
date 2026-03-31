# api/urls.py
from django.urls import path
from . import views

urlpatterns = [

    # Auth
    path('auth/register/', views.register,  name='register'),
    path('auth/login/',    views.login,     name='login'),
    path('auth/logout/',   views.logout,    name='logout'),

    # Writing features
    path('improve/',   views.improve_text,   name='improve'),
    path('email/',     views.generate_email, name='email'),
    path('summarize/', views.summarize,      name='summarize'),
    path('blog/',      views.generate_blog,  name='blog'),

    # Document 
    path('docs/',           views.documents,        name='documents'),
    path('docs/ask/',       views.ask_document,     name='ask-document'),
    path('docs/ask/v2/',    views.ask_document_v2,  name='ask-document-v2'),     
    path('docs/search/',    views.search_documents, name='search-documents'),
    path('docs/rewrite/',   views.rewrite_question, name='rewrite-question'),   
    path('docs/<int:pk>/',  views.delete_document,  name='delete-document'),

    # Chat
    path('conversations/',                views.conversations,       name='conversations'),
    path('conversations/<int:pk>/',       views.conversation_detail, name='conversation-detail'),
    path('chat/',                         views.chat,                name='chat'),
    path('conversations/<int:pk>/stats/', views.conversation_stats,  name='conversation-stats'),


    # Usage
    path('usage/', views.usage_stats, name='usage-stats'),
] 


'''
POST http://localhost:8000/api/auth/register/
Body: {"username": "testuser", "password": "testpass123", "email": "test@test.com"}
o/p: 
{
    "token": "893ec30502e205f7a250ab8b9f330f63afec4b4f",
    "username": "testuser",
    "message": "Account created successfully"
}

POST /api/improve/
Headers: Authorization: Token <your_token>
Body: {"text": "the meeting was not good and things went bad", "tone": "professional"}

o/p: 
{
    "original": "the meeting was not good and things went bad",
    "improved": "The meeting was unproductive, and subsequent developments were unfavorable.",
    "tone": "professional"
}

POST /api/email/
Body: {"bullets": "- project delayed\n- need 3 more days\n- will update daily", "tone": "professional"}
o/p: 
{
    "email": "Subject: Project Update and Timeline Adjustment\n\nDear [Recipient Name],\n\nI am writing to provide an important update regarding [Project Name/Our Current Project].\n\nDue to unforeseen circumstances, we anticipate a slight delay in the project's completion. We now estimate needing an additional three (3) business days to finalize all deliverables.\n\nTo ensure full transparency and keep you thoroughly informed, I will be providing daily updates on our progress until the project is successfully completed.\n\nWe appreciate your understanding and continued support.\n\nBest regards,\n[Your Name]",
    "tone": "professional"
}


POST /api/summarize/
Body: {"text": "<paste any long article>", "length": "short"}
o/p:
{
    "summary": "Saying \"no\" is the ultimate productivity hack, as it saves time and prevents overcommitment by avoiding unnecessary tasks and distractions. While social pressure often leads people to say \"yes,\" this creates \"time debt\" and diverts focus from important goals. Developing the skill to say \"no\" allows individuals to guard their time, focus on what truly matters, and ultimately achieve success.",
    "length": "short",
    "original_length": 7772,
    "summary_length": 393
}

POST /api/blog/
Body: {"topic": "Why Django is great for beginners", "keywords": "Python, web, REST API"}
o/p:
{
    "blog_post": "# Why Django is the Perfect Starting Point for Aspiring Web Developers\n\nStarting your journey into the vast and exciting world of web development can feel like standing at the edge of an ocean – exhilarating, but also a little overwhelming. With countless languages, frameworks, and tools vying for your attention, choosing where to begin can be the hardest part. If you're looking for a powerful, pragmatic, and incredibly beginner-friendly framework to kickstart your career, look no further than Django. This \"batteries-included\" framework, built on a highly accessible language, offers a smooth ramp-up for anyone eager to build robust web applications.\n\n## Built on Python – The Language of Choice for Beginners\n\nOne of Django's most significant advantages for newcomers is its foundation in **Python**. Often hailed as the most beginner-friendly programming language, Python boasts a syntax that is remarkably clean, readable, and intuitive, closely resembling natural language. This means that instead of grappling with complex semicolons, curly braces, and cryptic error messages, beginners can focus on understanding core programming concepts and application logic.\n\nIf you've already dipped your toes into Python, you're already well on your way to mastering Django. The transition is seamless because you're leveraging an existing skill set. Python's versatility extends beyond web development; it's used in data science, AI, automation, and more, making it an incredibly valuable language to learn overall. For Django beginners, Python's extensive ecosystem also means access to a vast array of libraries, a massive and supportive community, and countless tutorials and resources that translate directly into Django development. This strong linguistic bedrock significantly lowers the barrier to entry, allowing aspiring developers to build meaningful projects faster and with less frustration.\n\n## \"Batteries Included\" – A Comprehensive Framework for Quick Starts\n\nDjango proudly champions the \"batteries included\" philosophy, and this is a game-changer for beginners. What does it mean? It means that unlike many other frameworks where you might need to hunt for, integrate, and configure a multitude of third-party libraries for common functionalities, Django provides most of what you need right out of the box.\n\nThink about the essential components of almost any **web** application: user authentication (login, logout, password management), an administrative interface for managing data, an Object-Relational Mapper (ORM) for interacting with databases without writing raw SQL, URL routing, and a templating engine for rendering dynamic HTML. Django provides robust, production-ready solutions for all of these and more.\n\nFor a beginner, this is invaluable. It drastically reduces decision fatigue and the sheer complexity of setting up a project. You don't have to worry about which authentication library to choose, or how to integrate a database tool; Django has a cohesive, well-documented system already in place. The Django Admin interface, in particular, is a marvel for beginners. With just a few lines of code, you can get a fully functional, highly customizable backend dashboard to manage your application's data, allowing you to see your database models come to life almost instantly. This integrated approach allows beginners to focus on learning the framework's structure and building features, rather than getting bogged down in boilerplate setup.\n\n## Structured Learning and a Clear Path to Building Modern Web Applications\n\nDjango's opinionated nature, while sometimes seen as a constraint by seasoned developers, is a huge benefit for beginners. It encourages and enforces good software design patterns (specifically the Model-View-Template, or MVT, architecture) from the very start. This structured approach helps beginners develop good habits and understand how different parts of a web application interact, laying a solid foundation for future development.\n\nThe framework also boasts exceptional documentation, often cited as some of the best in the industry. It's comprehensive, well-organized, and provides clear examples, making it an invaluable resource for self-learning. Coupled with its massive and active community, beginners can easily find answers to questions, troubleshoot issues, and learn from experienced developers.\n\nMoreover, Django isn't just for traditional server-rendered web pages. It's perfectly suited for building modern **web** applications, including powerful backend services that drive single-page applications or mobile apps. With the widely used Django REST Framework (DRF), beginners can quickly learn to build robust **REST API**s. This is a crucial skill in today's interconnected world, where applications frequently communicate by exchanging data over APIs. DRF seamlessly integrates with Django's ORM and authentication system, making the process of exposing your application's data through a well-structured API surprisingly straightforward, even for those new to API development. Learning Django equips you not just with skills for traditional web development, but also for building the scalable, data-driven backends that power the most innovative applications today.\n\n## Conclusion\n\nDjango stands out as an exceptional choice for anyone embarking on their web development journey. Its foundation in the approachable **Python** language, its \"batteries included\" philosophy that simplifies project setup, and its structured approach to building applications provide a welcoming and effective learning environment. Whether you dream of building a dynamic **web** application from scratch or mastering the creation of robust **REST API**s, Django offers a clear, well-supported path to achieving your goals. Don't let the vastness of web development intimidate you. Dive into Django, explore its powerful features, and start building the amazing applications you envision today!",
    "topic": "Why Django is great for beginners",
    "word_count": 858
} 

POST /api/docs/
Body: form-data → file: <upload a PDF or TXT>
o/p: 
{
    "id": 1,
    "title": "Palash_Resume.pdf",
    "status": "ready",
    "chunk_count": 9,
    "uploaded_at": "2026-03-16T18:24:06.795006Z"
}

POST /api/docs/ask/
Body: {"document_id": 1, "question": "What is the main topic?"}
o/p: 
{
    "question": "What is the main topic?",
    "answer": "The main topic of the document is an individual's technical skills and project experience in software development, AI, and automation. This is evident from the \"TECHNICAL SKILLS\" section, the detailed project descriptions, and \"CERTIFICATIONS.\"",
    "sources": [
        {
            "text": "delivering responsive UI and seamless performance across device types.  \n▪ Built real-time attendance tracking system (Attendsync) with Firebase data ...",
            "page": 1,
            "score": 0.7402
        },
        {
            "text": "progression stages. \n▪ Action: Developed automated deep learning classification system using transfer learning with pre -trained \nResNet-18 CNN on the...",
            "page": 2,
            "score": 0.7561
        },
        {
            "text": "AI, Data Science, and AI Foundations. Skilled in Agile development, Git version control, and full app lifecycle from \narchitecture to Play Store deplo...",
            "page": 1,
            "score": 0.7684
        },
        {
            "text": "interactions. \n▪ Action: Designed and built a cross-platform Flutter app with color-fill, gesture recognition, undo/redo \nhistory, color palette manag...",
            "page": 2,
            "score": 0.7744
        }
    ],
    "document": "Palash_Resume.pdf"
}

POST /api/chat/
Body: {"message": "Hello! My name is Palas."}
o/p:
{
    "response": "Hello Palash! It's nice to meet you. How can I help you today?",
    "conversation_id": 2,
    "conversation_title": "Hello! My name is Palash."
}

POST /api/chat/
Body: {"message": "What is my name?", "conversation_id": 2}
o/p:
{
    "response": "Your name is Palash.",
    "conversation_id": 2,
    "conversation_title": "Hello! My name is Palash."
}

GET /api/usage/
o/p: 
{
    "total_requests": 8,
    "by_feature": {
        "chat": 3,
        "doc_qa": 1,
        "blog": 1,
        "summarize": 1,
        "email": 1,
        "improve": 1
    },
    "recent": [
        {
            "id": 8,
            "feature": "chat",
            "total_tokens": 0,
            "created_at": "2026-03-16T18:28:41.888478Z"
        },
        {
            "id": 7,
            "feature": "chat",
            "total_tokens": 0,
            "created_at": "2026-03-16T18:26:59.931057Z"
        },
        {
            "id": 6,
            "feature": "chat",
            "total_tokens": 0,
            "created_at": "2026-03-16T18:26:52.648280Z"
        },
        {
            "id": 5,
            "feature": "doc_qa",
            "total_tokens": 0,
            "created_at": "2026-03-16T18:25:36.482243Z"
        },
        {
            "id": 4,
            "feature": "blog",
            "total_tokens": 0,
            "created_at": "2026-03-16T18:13:52.164003Z"
        },
        {
            "id": 3,
            "feature": "summarize",
            "total_tokens": 0,
            "created_at": "2026-03-16T18:12:31.893101Z"
        },
        {
            "id": 2,
            "feature": "email",
            "total_tokens": 0,
            "created_at": "2026-03-16T18:04:30.898852Z"
        },
        {
            "id": 1,
            "feature": "improve",
            "total_tokens": 0,
            "created_at": "2026-03-16T17:58:14.673479Z"
        }
    ]
}
'''