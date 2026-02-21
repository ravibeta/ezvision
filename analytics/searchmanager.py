    async def create_agent(self):
        if self.search_info.agent_name:
            logger.info(f"Creating search agent named {self.search_info.agent_name}")

            async with self.search_info.create_search_index_client() as search_index_client:
                await search_index_client.create_or_update_agent(
                    agent=KnowledgeAgent(
                        name=self.search_info.agent_name,
                        target_indexes=[
                            KnowledgeAgentTargetIndex(
                                index_name=self.search_info.index_name, default_include_reference_source_data=True
                            )
                        ],
                        models=[
                            KnowledgeAgentAzureOpenAIModel(
                                azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                                    resource_url=self.search_info.azure_openai_endpoint,
                                    deployment_name=self.search_info.azure_openai_searchagent_deployment,
                                    model_name=self.search_info.azure_openai_searchagent_model,
                                )
                            )
                        ],
                        request_limits=KnowledgeAgentRequestLimits(
                            max_output_size=self.search_info.agent_max_output_tokens
                        ),
                    )
                )

            logger.info("Agent %s created successfully", self.search_info.agent_name)