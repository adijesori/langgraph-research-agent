import 'dotenv/config';
import { BedrockChat } from '@langchain/community/chat_models/bedrock';

import { StateGraph, Annotation, START } from '@langchain/langgraph';

import { ChatPromptTemplate } from '@langchain/core/prompts';

import { StringOutputParser } from '@langchain/core/output_parsers';

const StateAnnotation = Annotation.Root({
  topic: Annotation,
  research: Annotation,
});

function createResearchAgent() {
  const researcherPrompt = ChatPromptTemplate.fromTemplate(`
    You are a research agent that specializes in gathering information.
    
    Your task is to provide detailed information about the following topic:
    
    {topic}
    
    Provide clear, concise facts about this topic without opinion.
  `);

  const researchModel = new BedrockChat({
    model: 'anthropic.claude-v2',
    region: 'us-west-2',
    temperature: 0.3,
  });

  return researcherPrompt.pipe(researchModel).pipe(new StringOutputParser());
}

const researchAgent = createResearchAgent();

const workflow = new StateGraph(StateAnnotation);

workflow
  .addNode('researchNode', async ({ topic }) => {
    const researchResult = await researchAgent.invoke({ topic });
    return { research: researchResult };
  })
  .addEdge(START, 'researchNode');

export const graph = workflow.compile();
