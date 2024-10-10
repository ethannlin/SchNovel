'''
    File: generate_batch.py
    Description: This file generates the OpenAI batch json files for the Chain of Thought baseline tests.
'''

import json
from tqdm import tqdm

PROMPT = '''You will be provided 2 research paper’s title and abstract. Please determine which of the 2 articles is more novel. Follow these steps for evaluation.
            Step 1: Identity the problem and solution that the research paper attempts to solve.
            Step 2: Determine how unique the solution is given the current research landscape in 2024. Does the paper introduce a new idea, theory, or concept that has not been previously discussed in literature?
            Step 3: Determine how creative the solution is given the current research landscape in 2024. Does it apply a known idea in a completely new context or in a way that has not been done before?
            Step 4: Using the findings from steps 1-3, determine which paper is more novel.
            In your response please conclude with: "The more novel and impactful paper is [Paper X or Paper Y]'''

# Replace with filepath to [CATEGORY]'s json dataset
FILEPATH = ""

# Replace with destination filepath for batch files
BATCH_DST = ""

with open(FILEPATH, "r") as file:
    data = json.load(file)

data = data['data']

'''
    Creating the batch files

    The following code loops through the dataset and creates a batch file for each individual year and year gap.
    The code also swaps the position of the two papers. Using the provided dataset, there will be 25 batch files
    saved to your local directory each with 100 pairs (200 including swapped pairs)

    It includes two example pairs with reasoning for each prompt.
'''
for year, gap in data.items():
    for gap, papers in gap.items():
        batch = []
        counter = 1
        for i, paper in tqdm(enumerate(papers)):
            paper1_title = paper['paper1']['title']
            paper2_title = paper['paper2']['title']
            paper1_abstract = paper['paper1']['abstract']
            paper2_abstract = paper['paper2']['abstract']

            input = {
                "custom_id": f"{year}_{gap}_{counter}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": PROMPT
                        },
                        {
                            "role": "user",
                            "content": (
                                "Q: Which paper out of the two below is more novel?\n"
                                "Paper X Title: A bibliometric index based on the complete list of cited publications"
                                "Paper X Abstract: We propose a new index, the $j$-index, which is defined for an author as the\nsum of the square roots of the numbers of citations to each of the author's\npublications. The idea behind the $j$-index it to remedy a drawback of the\n$h$-index $-$ that the $h$-index does not take into account the full citation\nrecord of a researcher. The square root function is motivated by our desire to\navoid the possible bias that may occur with a simple sum when an author has\nseveral very highly cited papers. We compare the $j$-index to the $h$-index,\nthe $g$-index and the total citation count for three subject areas using\nseveral association measures.\n  Our results indicate that that the association between the $j$-index and the\nother indices varies according to the subject area. One explanation of this\nvariation may be due to the proportion of citations to publications of the\nresearcher that are in the $h$-core. The $j$-index is {\\em not} an $h$-index\nvariant, and as such is intended to complement rather than necessarily replace\nthe $h$-index and other bibliometric indicators, thus providing a more complete\npicture of a researcher's achievements.\n"
                                "Paper Y Title: Calorimeter shower superresolution"
                                "Paper Y Abstract Calorimeter shower simulation is a major bottleneck in the Large Hadron\nCollider computational pipeline. There have been recent efforts to employ\ndeep-generative surrogate models to overcome this challenge. However, many of\nbest performing models have training and generation times that do not scale\nwell to high-dimensional calorimeter showers. In this work, we introduce\nSuperCalo, a flow-based superresolution model, and demonstrate that\nhigh-dimensional fine-grained calorimeter showers can be quickly upsampled from\ncoarse-grained showers. This novel approach presents a way to reduce\ncomputational cost, memory requirements and generation time associated with\nfast calorimeter simulation models. Additionally, we show that the showers\nupsampled by SuperCalo possess a high degree of variation. This allows a large\nnumber of high-dimensional calorimeter showers to be upsampled from much fewer\ncoarse showers with high-fidelity, which results in additional reduction in\ngeneration time.\n"
                            )
                        },
                        {
                            "role": "assistant",
                            "content": "A: Let's think step by step.\nStep 1: Independent Evaluation\n\n**Paper X Evaluation:**\n- **Novelty of Methodology:** The introduction of the $j$-index as a new bibliometric index is somewhat novel as it proposes a different way to evaluate researchers' contributions beyond the established $h$-index. However, the idea of creating bibliometric indices is not new, and the method remains within the established framework of citation analysis.\n- **Surprisingness of Findings:** The findings indicating that the association between the $j$-index and other indices varies by subject area could be considered surprising, as it indicates a more nuanced understanding of citation metrics.\n- **Impact on Existing Knowledge:** While it challenges the limitations of the $h$-index, its impact is relatively low, functioning more as a complementary measure rather than offering profound new insights into citation dynamics.\n- **Potential for Future Research:** There may be implications for further refinement of bibliometric measures, but it does not open up substantial new directions for research.\n- **Relevance to 2024 Scientific Understanding:** Considering it's from 2013, it may not align closely with the current trends of advancements in citation analytics and bibliometric studies.\n\n**Score for Paper X:** 5/10\n- **Justification:** The novel proposal of the $j$-index provides a new perspective on citation measurement but remains within an established field of research without substantially altering the landscape.\n\n---\n\n**Paper Y Evaluation:**\n- **Novelty of Methodology:** The introduction of SuperCalo, a flow-based superresolution model for upsampling calorimeter showers, represents a significant shift in methodology that incorporates advanced deep learning approaches. The use of generative models in this context appears innovative.\n- **Surprisingness of Findings:** The ability of SuperCalo to drastically reduce computational costs while maintaining high fidelity in upsampled showers is unexpected and suggests a potentially revolutionary approach to simulation in high-dimensional spaces.\n- **Impact on Existing Knowledge:** This research addresses a major existing bottleneck in particle physics computations, indicating its high potential for practical application and substantial impact on future research in high-energy physics.\n- **Potential for Future Research:** The findings certainly open avenues for exploring other applications of flow-based models and could inspire further research into optimization and efficiency of simulations in related fields.\n- **Relevance to 2024 Scientific Understanding:** The paper aligns well with current trends emphasizing computational efficiency and advanced modeling in physics, thus reinforcing its relevance.\n\n**Score for Paper Y:** 9/10\n- **Justification:** The innovative use of deep generative models to tackle the specific bottleneck in calorimeter simulation is both groundbreaking and highly relevant, indicating strong potential implications for future research.\n\n---\n\nStep 2: Final Comparison\n\nLet's think step by step. \n\nWhen comparing Paper X and Paper Y, we observe that while Paper X introduces a new bibliometric index, it does so within a context that has not changed significantly since its publication. The methodology is a modest refinement of existing metrics, focusing on citation analysis, a relatively stable field. By contrast, Paper Y not only proposes a new methodology with the SuperCalo superresolution model but also addresses a pressing challenge in high-energy physics that is particularly relevant to current scientific discussion. \n\nThe evident shift from traditional bibliometric indexing in Paper X to an innovative machine learning approach in Paper Y indicates an advancement in methodology. The findings of Paper Y, highlighting significant improvements in computational efficiency and fidelity, showcase surprising results that have direct implications for research practices in a high-tech environment.\n\nMoreover, Paper Y opens up new directions for future research by introducing methods that could be applied to similar problems in computational physics, something that Paper X does not achieve to the same extent. \n\nIn terms of relevance, the deep learning technique introduced in Paper Y complements ongoing trends toward more efficient and high-performing computational methods, while Paper X's relevance is diminished by its older data and methodologies.\n\nThe more novel and impactful paper is Paper Y."
                        },
                        {
                            "role": "user",
                            "content": (
                                "Q: Which paper out of the two below is more novel?\n"
                                "Paper X Title: ATESA-B{\\AE}RT: A Heterogeneous Ensemble Learning Model for Aspect-Based\n  Sentiment Analysis"
                                "Paper X Abstract The increasing volume of online reviews has made possible the development of\nsentiment analysis models for determining the opinion of customers regarding\ndifferent products and services. Until now, sentiment analysis has proven to be\nan effective tool for determining the overall polarity of reviews. To improve\nthe granularity at the aspect level for a better understanding of the service\nor product, the task of aspect-based sentiment analysis aims to first identify\naspects and then determine the user's opinion about them. The complexity of\nthis task lies in the fact that the same review can present multiple aspects,\neach with its own polarity. Current solutions have poor performance on such\ndata. We address this problem by proposing ATESA-B{\\AE}RT, a heterogeneous\nensemble learning model for Aspect-Based Sentiment Analysis. Firstly, we divide\nour problem into two sub-tasks, i.e., Aspect Term Extraction and Aspect Term\nSentiment Analysis. Secondly, we use the \\textit{argmax} multi-class\nclassification on six transformers-based learners for each sub-task. Initial\nexperiments on two datasets prove that ATESA-B{\\AE}RT outperforms current\nstate-of-the-art solutions while solving the many aspects problem.\n"
                                "Paper Y Title: JRC-Names: A freely available, highly multilingual named entity resource"
                                "Paper Y Abstract: This paper describes a new, freely available, highly multilingual named\nentity resource for person and organisation names that has been compiled over\nseven years of large-scale multilingual news analysis combined with Wikipedia\nmining, resulting in 205,000 per-son and organisation names plus about the same\nnumber of spelling variants written in over 20 different scripts and in many\nmore languages. This resource, produced as part of the Europe Media Monitor\nactivity (EMM, http://emm.newsbrief.eu/overview.html), can be used for a number\nof purposes. These include improving name search in databases or on the\ninternet, seeding machine learning systems to learn named entity recognition\nrules, improve machine translation results, and more. We describe here how this\nresource was created; we give statistics on its current size; we address the\nissue of morphological inflection; and we give details regarding its\nfunctionality. Updates to this resource will be made available daily.\n"
                            )
                        },
                        {
                            "role": "assistant",
                            "content": "A: Let's think step by step.\nStep 1: Independent Evaluation\n\n**Paper X Evaluation**:\n- **Novelty of Methodology**: The paper proposes ATESA-BERT, a heterogeneous ensemble learning model, which divides the sentiment analysis task into two sub-tasks, showcasing an innovative approach to tackling aspect-based sentiment analysis.\n- **Surprisingness of Findings**: The assertion that ATESA-BERT outperforms state-of-the-art solutions is a significant claim, which suggests surprising effectiveness in resolving the complexities of multiple aspects in reviews.\n- **Impact on Existing Knowledge**: The paper aims to improve the granularity of sentiment analysis, potentially reshaping current methodologies in the field and enhancing understanding of user opinions.\n- **Potential for Future Research**: By showcasing improved performance on aspect-based sentiment analysis, the research opens pathways for further enhancements and exploration of ensemble learning techniques.\n- **Relevance to 2024 Scientific Understanding**: The timely exploration of sentiment analysis in the context of increasing online review volumes aligns well with contemporary trends in data analysis and machine learning.\n\n**Score for Paper X**: 9/10\n**Justification**: The methodology is innovative, and the findings suggest a significant potential to change sentiment analysis practices. The paper offers strong foundational work for future research, demonstrated by its claim of outperforming previous models.\n\n**Paper Y Evaluation**:\n- **Novelty of Methodology**: This paper introduces JRC-Names, presenting a multilingual resource derived from extensive news and Wikipedia mining. However, similar named entity resources have been developed previously.\n- **Surprisingness of Findings**: The findings are less surprising since they focus on the creation of a resource rather than novel algorithms or unexpected results.\n- **Impact on Existing Knowledge**: While the resource could indeed aid various applications, it does not fundamentally challenge or change the existing landscape of named entity recognition dramatically.\n- **Potential for Future Research**: The resource does provide utility and may serve as a foundation for further studies, but it does not inherently suggest new research avenues as directly as Paper X does.\n- **Relevance to 2024 Scientific Understanding**: While useful, the topic's relevance feels less pressing than methods related to dynamic sentiment analyses in the context of big data.\n\n**Score for Paper Y**: 6/10\n**Justification**: The introduction of a resource is beneficial, but it lacks the significant methodological novelty or surprising findings of Paper X.\n\nStep 2: Final Comparison\n\nLet's think step by step. The more recently published Paper X builds upon previous methods in sentiment analysis, specifically addressing the complex task of aspect-based sentiment analysis with its innovative ensemble learning model, ATESA-BERT. In contrast, Paper Y focuses on creating a multilingual named entity resource, a task that is somewhat common in the field of natural language processing.\n\nPaper X refines sentiment analysis methodologies by breaking the problem into more manageable sub-tasks, which is a novel approach that directly enhances the field's knowledge base. Furthermore, the abstract asserts that the model improves performance on datasets where other solutions have failed, stating, \"initial experiments prove that ATESA-BERT outperforms current state-of-the-art solutions,\" indicating groundbreaking effectiveness and potential for future research.\n\nConversely, Paper Y\u2019s contribution, while valuable, is more about the compilation and availability of data rather than introducing a new theoretical framework or challenging existing paradigms. Its abstract mentions, \"this resource can be used for a number of purposes,\" but it does not suggest any drastic improvements to named entity recognition methodologies akin to the innovations presented in Paper X.\n\nThe more novel and impactful paper is X."
                        },
                        {
                            "role": "user",
                            "content": (
                                "Q: Which paper out of the two below is more novel?\n"
                                f"Paper X Title: {paper1_title}\n"
                                f"Paper X Abstract: {paper1_abstract}\n"
                                f"Paper Y Title: {paper2_title}\n"
                                f"Paper Y Abstract: {paper2_abstract}"
                            )
                        },
                        {
                            "role": "assistant",
                            "content": "A: Let's think step by step.\n"
                        }
                    ]
                }
            }

            batch.append(input)
            counter += 1

            # Swap positions of input papers in the prompt
            input = {
                "custom_id": f"{year}_{gap}_{counter}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": PROMPT
                        },
                        {
                            "role": "user",
                            "content": (
                                "Q: Which paper out of the two below is more novel?\n"
                                "Paper X Title: A bibliometric index based on the complete list of cited publications"
                                "Paper X Abstract: We propose a new index, the $j$-index, which is defined for an author as the\nsum of the square roots of the numbers of citations to each of the author's\npublications. The idea behind the $j$-index it to remedy a drawback of the\n$h$-index $-$ that the $h$-index does not take into account the full citation\nrecord of a researcher. The square root function is motivated by our desire to\navoid the possible bias that may occur with a simple sum when an author has\nseveral very highly cited papers. We compare the $j$-index to the $h$-index,\nthe $g$-index and the total citation count for three subject areas using\nseveral association measures.\n  Our results indicate that that the association between the $j$-index and the\nother indices varies according to the subject area. One explanation of this\nvariation may be due to the proportion of citations to publications of the\nresearcher that are in the $h$-core. The $j$-index is {\\em not} an $h$-index\nvariant, and as such is intended to complement rather than necessarily replace\nthe $h$-index and other bibliometric indicators, thus providing a more complete\npicture of a researcher's achievements.\n"
                                "Paper Y Title: Calorimeter shower superresolution"
                                "Paper Y Abstract Calorimeter shower simulation is a major bottleneck in the Large Hadron\nCollider computational pipeline. There have been recent efforts to employ\ndeep-generative surrogate models to overcome this challenge. However, many of\nbest performing models have training and generation times that do not scale\nwell to high-dimensional calorimeter showers. In this work, we introduce\nSuperCalo, a flow-based superresolution model, and demonstrate that\nhigh-dimensional fine-grained calorimeter showers can be quickly upsampled from\ncoarse-grained showers. This novel approach presents a way to reduce\ncomputational cost, memory requirements and generation time associated with\nfast calorimeter simulation models. Additionally, we show that the showers\nupsampled by SuperCalo possess a high degree of variation. This allows a large\nnumber of high-dimensional calorimeter showers to be upsampled from much fewer\ncoarse showers with high-fidelity, which results in additional reduction in\ngeneration time.\n"
                            )
                        },
                        {
                            "role": "assistant",
                            "content": "A: Let's think step by step.\nStep 1: Independent Evaluation\n\n**Paper X Evaluation:**\n- **Novelty of Methodology:** The introduction of the $j$-index as a new bibliometric index is somewhat novel as it proposes a different way to evaluate researchers' contributions beyond the established $h$-index. However, the idea of creating bibliometric indices is not new, and the method remains within the established framework of citation analysis.\n- **Surprisingness of Findings:** The findings indicating that the association between the $j$-index and other indices varies by subject area could be considered surprising, as it indicates a more nuanced understanding of citation metrics.\n- **Impact on Existing Knowledge:** While it challenges the limitations of the $h$-index, its impact is relatively low, functioning more as a complementary measure rather than offering profound new insights into citation dynamics.\n- **Potential for Future Research:** There may be implications for further refinement of bibliometric measures, but it does not open up substantial new directions for research.\n- **Relevance to 2024 Scientific Understanding:** Considering it's from 2013, it may not align closely with the current trends of advancements in citation analytics and bibliometric studies.\n\n**Score for Paper X:** 5/10\n- **Justification:** The novel proposal of the $j$-index provides a new perspective on citation measurement but remains within an established field of research without substantially altering the landscape.\n\n---\n\n**Paper Y Evaluation:**\n- **Novelty of Methodology:** The introduction of SuperCalo, a flow-based superresolution model for upsampling calorimeter showers, represents a significant shift in methodology that incorporates advanced deep learning approaches. The use of generative models in this context appears innovative.\n- **Surprisingness of Findings:** The ability of SuperCalo to drastically reduce computational costs while maintaining high fidelity in upsampled showers is unexpected and suggests a potentially revolutionary approach to simulation in high-dimensional spaces.\n- **Impact on Existing Knowledge:** This research addresses a major existing bottleneck in particle physics computations, indicating its high potential for practical application and substantial impact on future research in high-energy physics.\n- **Potential for Future Research:** The findings certainly open avenues for exploring other applications of flow-based models and could inspire further research into optimization and efficiency of simulations in related fields.\n- **Relevance to 2024 Scientific Understanding:** The paper aligns well with current trends emphasizing computational efficiency and advanced modeling in physics, thus reinforcing its relevance.\n\n**Score for Paper Y:** 9/10\n- **Justification:** The innovative use of deep generative models to tackle the specific bottleneck in calorimeter simulation is both groundbreaking and highly relevant, indicating strong potential implications for future research.\n\n---\n\nStep 2: Final Comparison\n\nLet's think step by step. \n\nWhen comparing Paper X and Paper Y, we observe that while Paper X introduces a new bibliometric index, it does so within a context that has not changed significantly since its publication. The methodology is a modest refinement of existing metrics, focusing on citation analysis, a relatively stable field. By contrast, Paper Y not only proposes a new methodology with the SuperCalo superresolution model but also addresses a pressing challenge in high-energy physics that is particularly relevant to current scientific discussion. \n\nThe evident shift from traditional bibliometric indexing in Paper X to an innovative machine learning approach in Paper Y indicates an advancement in methodology. The findings of Paper Y, highlighting significant improvements in computational efficiency and fidelity, showcase surprising results that have direct implications for research practices in a high-tech environment.\n\nMoreover, Paper Y opens up new directions for future research by introducing methods that could be applied to similar problems in computational physics, something that Paper X does not achieve to the same extent. \n\nIn terms of relevance, the deep learning technique introduced in Paper Y complements ongoing trends toward more efficient and high-performing computational methods, while Paper X's relevance is diminished by its older data and methodologies.\n\nThe more novel and impactful paper is Paper Y."
                        },
                        {
                            "role": "user",
                            "content": (
                                "Q: Which paper out of the two below is more novel?\n"
                                "Paper X Title: ATESA-B{\\AE}RT: A Heterogeneous Ensemble Learning Model for Aspect-Based\n  Sentiment Analysis"
                                "Paper X Abstract The increasing volume of online reviews has made possible the development of\nsentiment analysis models for determining the opinion of customers regarding\ndifferent products and services. Until now, sentiment analysis has proven to be\nan effective tool for determining the overall polarity of reviews. To improve\nthe granularity at the aspect level for a better understanding of the service\nor product, the task of aspect-based sentiment analysis aims to first identify\naspects and then determine the user's opinion about them. The complexity of\nthis task lies in the fact that the same review can present multiple aspects,\neach with its own polarity. Current solutions have poor performance on such\ndata. We address this problem by proposing ATESA-B{\\AE}RT, a heterogeneous\nensemble learning model for Aspect-Based Sentiment Analysis. Firstly, we divide\nour problem into two sub-tasks, i.e., Aspect Term Extraction and Aspect Term\nSentiment Analysis. Secondly, we use the \\textit{argmax} multi-class\nclassification on six transformers-based learners for each sub-task. Initial\nexperiments on two datasets prove that ATESA-B{\\AE}RT outperforms current\nstate-of-the-art solutions while solving the many aspects problem.\n"
                                "Paper Y Title: JRC-Names: A freely available, highly multilingual named entity resource"
                                "Paper Y Abstract: This paper describes a new, freely available, highly multilingual named\nentity resource for person and organisation names that has been compiled over\nseven years of large-scale multilingual news analysis combined with Wikipedia\nmining, resulting in 205,000 per-son and organisation names plus about the same\nnumber of spelling variants written in over 20 different scripts and in many\nmore languages. This resource, produced as part of the Europe Media Monitor\nactivity (EMM, http://emm.newsbrief.eu/overview.html), can be used for a number\nof purposes. These include improving name search in databases or on the\ninternet, seeding machine learning systems to learn named entity recognition\nrules, improve machine translation results, and more. We describe here how this\nresource was created; we give statistics on its current size; we address the\nissue of morphological inflection; and we give details regarding its\nfunctionality. Updates to this resource will be made available daily.\n"
                            )
                        },
                        {
                            "role": "assistant",
                            "content": "A: Let's think step by step.\nStep 1: Independent Evaluation\n\n**Paper X Evaluation**:\n- **Novelty of Methodology**: The paper proposes ATESA-BERT, a heterogeneous ensemble learning model, which divides the sentiment analysis task into two sub-tasks, showcasing an innovative approach to tackling aspect-based sentiment analysis.\n- **Surprisingness of Findings**: The assertion that ATESA-BERT outperforms state-of-the-art solutions is a significant claim, which suggests surprising effectiveness in resolving the complexities of multiple aspects in reviews.\n- **Impact on Existing Knowledge**: The paper aims to improve the granularity of sentiment analysis, potentially reshaping current methodologies in the field and enhancing understanding of user opinions.\n- **Potential for Future Research**: By showcasing improved performance on aspect-based sentiment analysis, the research opens pathways for further enhancements and exploration of ensemble learning techniques.\n- **Relevance to 2024 Scientific Understanding**: The timely exploration of sentiment analysis in the context of increasing online review volumes aligns well with contemporary trends in data analysis and machine learning.\n\n**Score for Paper X**: 9/10\n**Justification**: The methodology is innovative, and the findings suggest a significant potential to change sentiment analysis practices. The paper offers strong foundational work for future research, demonstrated by its claim of outperforming previous models.\n\n**Paper Y Evaluation**:\n- **Novelty of Methodology**: This paper introduces JRC-Names, presenting a multilingual resource derived from extensive news and Wikipedia mining. However, similar named entity resources have been developed previously.\n- **Surprisingness of Findings**: The findings are less surprising since they focus on the creation of a resource rather than novel algorithms or unexpected results.\n- **Impact on Existing Knowledge**: While the resource could indeed aid various applications, it does not fundamentally challenge or change the existing landscape of named entity recognition dramatically.\n- **Potential for Future Research**: The resource does provide utility and may serve as a foundation for further studies, but it does not inherently suggest new research avenues as directly as Paper X does.\n- **Relevance to 2024 Scientific Understanding**: While useful, the topic's relevance feels less pressing than methods related to dynamic sentiment analyses in the context of big data.\n\n**Score for Paper Y**: 6/10\n**Justification**: The introduction of a resource is beneficial, but it lacks the significant methodological novelty or surprising findings of Paper X.\n\nStep 2: Final Comparison\n\nLet's think step by step. The more recently published Paper X builds upon previous methods in sentiment analysis, specifically addressing the complex task of aspect-based sentiment analysis with its innovative ensemble learning model, ATESA-BERT. In contrast, Paper Y focuses on creating a multilingual named entity resource, a task that is somewhat common in the field of natural language processing.\n\nPaper X refines sentiment analysis methodologies by breaking the problem into more manageable sub-tasks, which is a novel approach that directly enhances the field's knowledge base. Furthermore, the abstract asserts that the model improves performance on datasets where other solutions have failed, stating, \"initial experiments prove that ATESA-BERT outperforms current state-of-the-art solutions,\" indicating groundbreaking effectiveness and potential for future research.\n\nConversely, Paper Y\u2019s contribution, while valuable, is more about the compilation and availability of data rather than introducing a new theoretical framework or challenging existing paradigms. Its abstract mentions, \"this resource can be used for a number of purposes,\" but it does not suggest any drastic improvements to named entity recognition methodologies akin to the innovations presented in Paper X.\n\nThe more novel and impactful paper is X."
                        },
                        {
                            "role": "user",
                            "content": (
                                "Q: Which paper out of the two below is more novel?\n"
                                f"Paper X Title: {paper2_title}\n"
                                f"Paper X Abstract: {paper2_abstract}\n"
                                f"Paper Y Title: {paper1_title}\n"
                                f"Paper Y Abstract: {paper1_abstract}"
                            )
                        },
                        {
                            "role": "assistant",
                            "content": "A: Let's think step by step.\n"
                        }
                    ]
                }
            }

            batch.append(input)
            counter += 1

        # Write batch to after every gap
        with open(f"{BATCH_DST}/{year}_{gap}_batch.json", "w") as file:
            for item in batch:
                json.dump(item, file)
                file.write("\n")